# =================================
# using for initialize data sets
# =================================
import sample.jieba as jieba
import numpy as np
import progressbar
import csv
from math import floor
import pandas as pd
import file_process as fp
import tensorflow as tf
import os
from hanziconv import HanziConv
import functools
tf.flags.DEFINE_integer(
    "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string(
    "input_dir", os.path.abspath("./data"),
    "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
    "output_dir", os.path.abspath("./data"),
    "Output directory for TFrEcord files (default = './data')")
tf.flags.DEFINE_integer(
    "distraction_num", 9,
    "Output directory for TFrEcord files (default = './data')")
FLAGS = tf.flags.FLAGS


class Dataset:
    # private data
    _stopwordset = ''
    _splitsymbol = ''

    _counter = 0
    _bar = progressbar.ProgressBar(maxval=100)
    raw_data = []
    # word_data = []
    # id_data = []
    # _vec_data = []
    _data_len = 0

    _split_scope_sum = 0
    _split_scope_1 = 0
    _split_scope_2 = 0

    # public data
    train_data = []
    valid_data = []
    test_data = []

    id2vec_lookup_list = []
    word2id_lookup_list = {}
    # path
    _userdict = ''

    def __init__(self, *, filename, splitsymbol, word2id_file, id2vec_file, user_dict, stopword_dict, prop):
        # ray.init(redis_address="127.0.0.1:35247")
        self._splitsymbol = splitsymbol

        # 初始化停用词表
        if stopword_dict is not []:
            self.set_stopword(stopword_dict)

        # 初始化自定义词表
        if user_dict is not []:
            self._userdict = user_dict

        # 加载词向量和id查找表
        self.word2id_lookup_list = fp.load_obj(word2id_file)
        length = len(self.word2id_lookup_list)
        self.id2vec_lookup_list = np.load(id2vec_file)

        # <unknown> : set values of vector as all 0
        self.word2id_lookup_list.update({'<unknown>': length})
        self.id2vec_lookup_list = np.append(
            self.id2vec_lookup_list, [np.zeros(300)], axis=0)

        jieba.load_userdict(self._userdict)

        # 导入数据集
        # 读取csv, 创建dataframe
        print('导入数据集')
        with open(filename, "r") as csvFile:
            reader = csv.reader(csvFile)
            self.raw_data = [{
                'question': row[0], 'answer': row[1]} for row in reader]
            csvFile.close()
        self._data_len = len(self.raw_data)

        # 切分数据集
        self.split_data_set(prop, self.raw_data, self._data_len)

    def split_data_set(self, prop, data, length):
        """
        分配数据集
            :param self:
            :param prop:
        """  # split data
        # split=[]
        self._split_scope_sum = sum(prop)
        self._split_scope_1 = prop[0] / self._split_scope_sum
        self._split_scope_2 = prop[1] / self._split_scope_sum

        train_data_begin = 0
        train_data_end = train_data_begin + floor(self._split_scope_1 * length)
        valid_data_begin = train_data_end
        valid_data_end = valid_data_begin + floor(self._split_scope_2 * length)
        test_data_begin = valid_data_end
        test_data_end = length

        self.train_data = data[train_data_begin:train_data_end]
        self.test_data = data[valid_data_begin:valid_data_end]
        self.valid_data = data[test_data_begin:test_data_end]

        def map_neg_utterance(i):
            try:
                utterance_index = np.random.random_integers(
                    1, length)  # 避免撞上自己
                utterance_index = utterance_index % length
                utterance = self.raw_data[utterance_index]['answer']
                return utterance
            except Exception as e:
                print(e)

        # add neg sample into training data
        def map_train_data(data):
            data['label'] = 0
            return data

        pos_train_data = list(map(map_train_data, self.train_data))
        neg_utterance = list(
            map(map_neg_utterance, self.train_data))
        neg_train_data = [{
            'question': self.train_data[i]['question'], 'answer': neg_utterance[i], 'label': 1} for i in range(len(self.train_data))]
        self.train_data = pos_train_data + neg_train_data

        # Distractor sequences
        def map_test_data(_data):
            for i in range(FLAGS.distraction_num):
                try:
                    dis_key = "distractor_{}".format(i)
                    # Distractor Text Feature
                    dis_text = map_neg_utterance(i)
                    _data[dis_key] = dis_text
                except Exception as e:
                    print(e)

            return _data

        self.test_data = list(map(map_test_data, self.test_data))
        self.valid_data = list(map(map_test_data, self.valid_data))

    def set_stopword(self, files):
        """
        load stop words
        """
        try:
            stopwords = ''
            for item in files:
                stopwords = ''.join(stopwords + '\n' +
                                    fp.readfile(item))
            stopwordslist = stopwords.split('\n')
            not_stopword = set(['', '\n'])
            self._stopwordset = set(stopwordslist)
            self._stopwordset = self._stopwordset - not_stopword
        except Exception as e:
            print(e)

    def movestopwords(self, sentence):
        '''
        remove stop words
        '''
        try:
            def is_stopwords(word):
                if (word != '\t' and '\n') and (word not in self._stopwordset):
                    return word and word.strip()

            res = list(filter(is_stopwords, sentence))
            return res
        except Exception as e:
            print(e)

    def chinese_tokenizer(self, documents):
        for document in documents:
            # 繁体转简体
            text = HanziConv.toSimplified(document)
            # 英文转小写
            text = jieba.lcut(text)
            # 去除停用词
            if self._stopwordset:
                text = self.movestopwords(text)
            # 分词
            yield text

    def create_csv_iter(self, filename):
        """
        Returns an iterator over a CSV file. Skips the header.
        """
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the header
            next(reader)
            for row in reader:
                yield row

    def create_vocab(self, input_iter, min_frequency):
        """
        Creates and returns a VocabularyProcessor object with the vocabulary
        for the input iterator.
        """
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            FLAGS.max_sentence_len,
            min_frequency=min_frequency,
            tokenizer_fn=self.chinese_tokenizer,
            # vocabulary=self.word2id_lookup_list
        )
        vocab_processor.fit(input_iter)
        return vocab_processor

    def transform_sentence(self, sequence, vocab_processor):
        """
        Maps a single sentence into the integer vocabulary. Returns a python array.
        """
        return next(vocab_processor.transform([sequence])).tolist()

    def create_text_sequence_feature(self, fl, sentence, sentence_len, vocab):
        """
        Writes a sentence to FeatureList protocol buffer
        """
        sentence_transformed = self.transform_sentence(sentence, vocab)
        for word_id in sentence_transformed:
            fl.feature.add().int64_list.value.extend([word_id])
        return fl

    def create_example_train(self, row, vocab):
        """
        Creates a training example for the Ubuntu Dialog Corpus dataset.
        Returnsthe a tensorflow.Example Protocol Buffer object.
        """
        context, utterance, label = row
        context_transformed = self.transform_sentence(context, vocab)
        utterance_transformed = self.transform_sentence(utterance, vocab)
        context_len = len(next(vocab._tokenizer([context])))
        utterance_len = len(next(vocab._tokenizer([utterance])))
        label = int(float(label))

        # New Example
        example = tf.train.Example()
        example.features.feature["context"].int64_list.value.extend(
            context_transformed)
        example.features.feature["utterance"].int64_list.value.extend(
            utterance_transformed)
        example.features.feature["context_len"].int64_list.value.extend([
                                                                        context_len])
        example.features.feature["utterance_len"].int64_list.value.extend([
            utterance_len])
        example.features.feature["label"].int64_list.value.extend([label])
        return example

    def create_example_test(self, row, vocab):
        """
        Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
        Returnsthe a tensorflow.Example Protocol Buffer object.
        """
        context, utterance = row[:2]
        distractors = row[2:]
        context_len = len(next(vocab._tokenizer([context])))
        utterance_len = len(next(vocab._tokenizer([utterance])))
        context_transformed = self.transform_sentence(context, vocab)
        utterance_transformed = self.transform_sentence(utterance, vocab)

        # New Example
        example = tf.train.Example()
        example.features.feature["context"].int64_list.value.extend(
            context_transformed)
        example.features.feature["utterance"].int64_list.value.extend(
            utterance_transformed)
        example.features.feature["context_len"].int64_list.value.extend([
                                                                        context_len])
        example.features.feature["utterance_len"].int64_list.value.extend([
            utterance_len])

        # Distractor sequences
        for i, distractor in enumerate(distractors):
            dis_key = "distractor_{}".format(i)
            dis_len_key = "distractor_{}_len".format(i)
            # Distractor Length Feature
            dis_len = len(next(vocab._tokenizer([distractor])))
            example.features.feature[dis_len_key].int64_list.value.extend([
                dis_len])
            # Distractor Text Feature
            dis_transformed = self.transform_sentence(distractor, vocab)
            example.features.feature[dis_key].int64_list.value.extend(
                dis_transformed)
        return example

    def create_tfrecords_file(self, data, output_filename, example_fn):
        """
        Creates a TFRecords file for the given input data and
        example transofmration function
        """
        with tf.python_io.TFRecordWriter(output_filename) as writer:
            print("Creating TFRecords file at {}...".format(output_filename))

            examples = fp.pool_map(lambda row: example_fn(
                row).SerializeToString(), data)

            print("Wrote to {}".format(output_filename))
            for item in examples:
                writer.write(item)

    def write_vocabulary(self, vocab_processor, outfile):
        """
        Writes the vocabulary to a file, one word per line.
        """
        vocab_size = len(vocab_processor.vocabulary_)
        with open(outfile, "w") as vocabfile:
            for _id in range(vocab_size):
                word = vocab_processor.vocabulary_._reverse_mapping[_id]
                vocabfile.write(word + "\n")
        print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
    dataset = Dataset(filename="corpus/corpus.csv",
                      splitsymbol='<POS>',
                      word2id_file='sample/word2vec/id2word.pkl',
                      id2vec_file='sample/word2vec/vec.npz.npy',
                      user_dict="dict/自定义词典.txt",
                      stopword_dict=[
                          'dict/哈工大停用词表.txt',
                          #    'dict/中文停用词.txt',
                          'dict/自定义停用词.txt'],
                      prop=[0.6, 0.2, 0.2])

    # fp.save_obj(dataset, 'save.pkl')
    # print("Read Dataset...")
    # dataset = TF_Dataset(pkl_filename='save.pkl')

    print("Creating vocabulary...")
    input_iter = [x['question']+' '+x['answer'] for x in dataset.raw_data]
    vocab = dataset.create_vocab(
        input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    dataset.write_vocabulary(
        vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

    # Save vocab processor
    vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

    # Create tfrecords
    print("Creating validation tfrecords...")
    input = [list(x.values()) for x in dataset.valid_data]
    dataset.create_tfrecords_file(input,
                                  output_filename=os.path.join(
                                      FLAGS.output_dir, "validation.tfrecords"),
                                  example_fn=functools.partial(dataset.create_example_test, vocab=vocab))

    print("Creating test tfrecords...")
    input = [list(x.values()) for x in dataset.test_data]
    dataset.create_tfrecords_file(input,
                                  output_filename=os.path.join(
                                      FLAGS.output_dir, "test.tfrecords"),
                                  example_fn=functools.partial(dataset.create_example_test, vocab=vocab))  # 固定函数变量

    print("Creating valitraindation tfrecords...")
    input = [list(x.values()) for x in dataset.train_data]
    dataset.create_tfrecords_file(input,
                                  output_filename=os.path.join(
                                      FLAGS.output_dir, "train.tfrecords"),
                                  example_fn=functools.partial(dataset.create_example_train, vocab=vocab))

    pass
