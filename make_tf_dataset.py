# encoding=utf-8

import file_process as fp
import numpy as np
import tensorflow as tf
from prepare_data import Dataset  # 声明定义域


class TF_Dataset():
    """docstring for TF_Dataset."""

    dateset = []

    def __init__(self, *, pkl_filename):
        self.dateset = fp.load_obj(pkl_filename)

    def create_example_train(self):
        """
        Creates a training example for the Ubuntu Dialog Corpus dataset.
        Returnsthe a tensorflow.Example Protocol Buffer object.
        """
        data = self.dateset.train_data
        length = len(data)
        pos_context = [x[3] for x in data]
        pos_utterance = [x[3] for x in data]
        pos_label = np.ones(length)

        def map_neg_utterance(i, x):
            utterance_index = np.random.random_integers(length) + i
            utterance_index = utterance_index if utterance_index < length else utterance_index-length
            utterance = data[utterance_index][4]

        neg_context = pos_context
        neg_utterance = list(map(map_neg_utterance,
                            enumerate(data)))
        neg_label = np.zeros(length)

        context = [pos_context, neg_context]
        utterance = [pos_utterance, neg_utterance]
        label = [pos_label, neg_label]
        # context_transformed = transform_sentence(context, vocab)
        # utterance_transformed = transform_sentence(utterance, vocab)
        context_len = list(map(len, context))
        utterance_len = list(map(len, utterance))

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
            dis_transformed = transform_sentence(distractor, vocab)
            example.features.feature[dis_key].int64_list.value.extend(
                dis_transformed)
        return example

    def create_tfrecords_file(self, input_filename, output_filename, example_fn):
        """
        Creates a TFRecords file for the given input data and
        example transofmration function
        """
        writer = tf.python_io.TFRecordWriter(output_filename)
        print("Creating TFRecords file at {}...".format(output_filename))
        for i, row in enumerate(create_csv_iter(input_filename)):
            x = example_fn(row)
            writer.write(x.SerializeToString())
        writer.close()
        print("Wrote to {}".format(output_filename))


if __name__ == "__main__":
    dataset = TF_Dataset(pkl_filename='save.pkl')
    dataset.create_example_train()
    pass
