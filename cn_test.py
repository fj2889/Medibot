import os
import time
import itertools
import sys
import tensorflow as tf
import cn_model
import cn_hparams
import cn_metrics
import cn_inputs
from models.model import dual_encoder_model
from models import model

tf.flags.DEFINE_string("test_file", "./data/test.tfrecords", "Path of test data in TFRecords format")
tf.flags.DEFINE_string("model_dir", "./runs/RNN_CNN_MaxPooling", "Directory to load model checkpoints from")
tf.flags.DEFINE_integer("test_batch_size", 16, "Batch size for testing")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

tf.logging.set_verbosity(FLAGS.loglevel)

if __name__ == "__main__":
  hparams = cn_hparams.create_hparams()
  model_fn = cn_model.create_model_fn(
          hparams,
          model_impl=dual_encoder_model,
          model_fun=model.RNN_CNN_MaxPooling,
          RNNInit=tf.nn.rnn_cell.LSTMCell,
          is_bidirection=True
      )
  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=tf.contrib.learn.RunConfig())

  input_fn_test = cn_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[FLAGS.test_file],
    batch_size=FLAGS.test_batch_size,
    num_epochs=1)

  eval_metrics = cn_metrics.create_evaluation_metrics()
  estimator.evaluate(input_fn=input_fn_test, steps=None, metrics=eval_metrics)
