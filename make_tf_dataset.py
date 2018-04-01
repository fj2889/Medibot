# encoding=utf-8

import file_process as fp
import numpy as np
import tensorflow as tf
import os
from prepare_data import Dataset  # 声明定义域

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
    "distraction_num", 9, 'number of distractions')
FLAGS = tf.flags.FLAGS


class TF_Dataset():
    """docstring for TF_Dataset."""

    dataset = []

    def __init__(self, *, pkl_filename):
        self.dataset = fp.load_obj(pkl_filename)

    def create_vocab(self, input_iter, min_frequency):
        """
        Creates and returns a VocabularyProcessor object with the vocabulary
        for the input iterator.
        """
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            FLAGS.max_sentence_len,
            min_frequency=min_frequency,
            tokenizer_fn=self.dataset.chinese_tokenizer)
        vocab_processor.fit(input_iter)
        return vocab_processor

    def write_vocabulary(self, vocab_processor, outfile):
        """
        Writes the vocabulary to a file, one word per line.
        """
        vocab_size = len(vocab_processor.vocabulary_)
        with open(outfile, "w") as vocabfile:
            for id in range(vocab_size):
                word = vocab_processor.vocabulary_._reverse_mapping[id]
                vocabfile.write(word + "\n")
        print("Saved vocabulary to {}".format(outfile))

    def create_example(self, output_filename, mode):
        """
        Creates a training example for the Ubuntu Dialog Corpus dataset.
        Returnsthe a tensorflow.Example Protocol Buffer object.
        """
        data = []
        if mode == 'train':
            data = self.dataset.train_data
        elif mode == 'valid':
            data = self.dataset.valid_data
        elif mode == 'test':
            data = self.dataset.test_data
        length = len(data)

        def map_neg_utterance(i):
            try:
                utterance_index = np.random.random_integers(
                    1, length) + i  # 避免撞上自己
                utterance_index = utterance_index if utterance_index < length else utterance_index - length
                utterance = self.dataset.word_data[utterance_index][4]
                return utterance
            except Exception as e:
                print(e)

        if mode == 'train':
            def map_record_train(input):
                context, utterance, context_len, utterance_len, label = input
                example = tf.train.Example()
                example.features.feature["context"].int64_list.value.extend(
                    context)
                example.features.feature["utterance"].int64_list.value.extend(
                    utterance)
                example.features.feature["context_len"].int64_list.value.extend([
                    context_len])
                example.features.feature["utterance_len"].int64_list.value.extend([
                    utterance_len])
                example.features.feature["label"].int64_list.value.extend([
                    int(label)])
                return example.SerializeToString()

            pos_context = [x[3] for x in data]
            pos_utterance = [x[4] for x in data]
            pos_label = np.ones(length)

            neg_context = pos_context
            neg_utterance = list(map(map_neg_utterance, range(length)))
            neg_label = np.zeros(length)

            context = pos_context + neg_context
            utterance = pos_utterance + neg_utterance
            label = np.concatenate((pos_label, neg_label))
            # context_transformed = transform_sentence(context, vocab)
            # utterance_transformed = transform_sentence(utterance, vocab)
            context_len = list(map(len, context))
            utterance_len = list(map(len, utterance))
            _range = range(len(context))
            param = ([context[i], utterance[i], context_len[i], utterance_len[i], label[i]]
                     for i in _range)
            examples = fp.pool_map(map_record_train, param)
        else:
            def map_record_test(input):
                context, context_len, utterance,  utterance_len = input
                # New Example
                example = tf.train.Example()
                example.features.feature["context"].int64_list.value.extend(
                    context)
                example.features.feature["utterance"].int64_list.value.extend(
                    utterance)
                example.features.feature["context_len"].int64_list.value.extend([
                    context_len])
                example.features.feature["utterance_len"].int64_list.value.extend([
                    utterance_len])

                # Distractor sequences
                for i in range(FLAGS.distraction_num):
                    dis_key = "distractor_{}".format(i)
                    dis_len_key = "distractor_{}_len".format(i)
                    # Distractor Text Feature
                    dis_text = map_neg_utterance(i)
                    # Distractor Length Feature
                    dis_len = len(dis_text)

                    example.features.feature[dis_len_key].int64_list.value.extend(
                        dis_text)
                    example.features.feature[dis_key].int64_list.value.extend(
                        [dis_len])
                return example.SerializeToString()

            param = ([x[3], len(x[3]), x[4], len(x[4])]
                     for x in data)  # context+context_len+ utterance +utterance_len
            examples = fp.pool_map(map_record_test, param)
        # New Example
        print("Creating TFRecords file at {}...".format(output_filename))
        with tf.python_io.TFRecordWriter(output_filename) as writer:
            for item in examples:
                writer.write(item)
        print("Wrote to {}".format(output_filename))


if __name__ == "__main__":
    print("Read Dataset...")
    dataset = TF_Dataset(pkl_filename='save.pkl')

    print("Creating vocabulary...")
    input_iter = [x[1]+x[2] for x in dataset.dataset.word_data]
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
    dataset.create_example(
        output_filename=os.path.join(
            FLAGS.output_dir, "validation.tfrecords"),
        mode='valid')

    print("Creating test tfrecords...")
    dataset.create_example(
        output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
        mode='test')

    print("Creating valitraindation tfrecords...")
    dataset.create_example(
        output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
        mode='train')

    pass
