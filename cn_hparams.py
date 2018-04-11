import tensorflow as tf
import os
from collections import namedtuple
# 1495
# Model Parameters
tf.flags.DEFINE_integer(
    "vocab_size",
    3398,
    "The size of the vocabulary. Only change this if you changed the preprocessing")
# Model Parameters
'''tf.flags.DEFINE_integer("embedding_dim", 300,
                        "Dimensionality of the embeddings")'''
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160,
                        "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80,
                        "Truncate utterance to this length")





# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam",
                       "Optimizer Name (Adam, Adagrad, etc)")

tf.flags.DEFINE_boolean("old_data", False,
                        "decide if use the old data")
if tf.flags.FLAGS.old_data:
    print("using old data/word vector/vocabulary/vocab_processor")
    tf.flags.DEFINE_integer("embedding_dim", 100,
                            "Dimensionality of the embeddings")
    # /data文件夹
    tf.flags.DEFINE_string("input_dir", "./old_data",
                           "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")

    # Pre-trained embeddings
    # "./data/glove.6B.100d.txt"
    # ./data/vocabulary.txt
    tf.flags.DEFINE_string("word2vec_path", "./old_data/glove.6B.100d.txt",
                           "Path to pre-trained Glove vectors")

    tf.flags.DEFINE_string("vocab_path", './old_data/vocabulary.txt',
                           "Path to vocabulary.txt file")
    tf.flags.DEFINE_string("vocab_processor_file", "./old_data/vocab_processor.bin", "Saved vocabulary processor file")
else:
    print("using new data/word vector/vocabulary/vocab_processor")
    tf.flags.DEFINE_integer("embedding_dim", 300,
                            "Dimensionality of the embeddings")
    # /data文件夹
    tf.flags.DEFINE_string("input_dir", "./data",
                           "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
    tf.flags.DEFINE_string("word2vec_path", 'word2vec/word2vec.npy',
                           "Path to dataset.pkl file")
    tf.flags.DEFINE_string("vocab_path", './data/vocabulary.txt',
                           "Path to vocabulary.txt file")
    tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")

tf.flags.DEFINE_string("RNN_CNN_MaxPooling_model_dir", 'runs/RNN_CNN_MaxPooling',
                       "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_string("RNN_MaxPooling_model_dir", 'runs/RNN_MaxPooling',
                       "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_string("RNN_model_dir", 'runs/RNN',
                       "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None,
                        "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 105,
                        "Evaluate after this many train steps")


tf.flags.DEFINE_integer("min_word_frequency", 5,
                        "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

# tf.flags.DEFINE_string(
#     "input_dir", os.path.abspath("./data"),
#     "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
    "output_dir", os.path.abspath("./data"),
    "Output directory for TFrEcord files (default = './data')")
tf.flags.DEFINE_integer(
    "distraction_num", 9,
    "Output directory for TFrEcord files (default = './data')")


FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "embedding_dim",
        "eval_batch_size",
        "learning_rate",
        "max_context_len",
        "max_utterance_len",
        "optimizer",
        "rnn_dim",
        "vocab_size",
        "vocab_path",
        "word2vec_path",

    ])


def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        max_context_len=FLAGS.max_context_len,
        max_utterance_len=FLAGS.max_utterance_len,
        vocab_path=FLAGS.vocab_path,
        word2vec_path=FLAGS.word2vec_path,
        rnn_dim=FLAGS.rnn_dim)
