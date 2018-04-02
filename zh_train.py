import os
import time
import itertools
import tensorflow as tf
import zh_model
import zh_hparams
import zh_metrics
import zh_inputs
from models.dual_encoder import dual_encoder_model
from tensorflow.contrib.learn import Estimator
from models import model
import functools

import make_data
from make_data import Dataset  # 声明定义域

FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

if FLAGS.model_dir:
  MODEL_DIR = FLAGS.model_dir
else:
  MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_argv):
  hparams = zh_hparams.create_hparams()
  #model_fun=[2,3,4,5],30
  model_fn = zh_model.create_model_fn(
    hparams,
    model_impl=dual_encoder_model,
    #model_fun=functools.partial(model.RNN_CNN_MaxPooling,filtersizes=[2,3,4,5],num_filters=30),
    model_fun=model.RNN_CNN_MaxPooling,
    RNNInit=tf.nn.rnn_cell.LSTMCell,
    isBiDirection=True)

  estimator = Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig())
  #tf.contrib.learn.RunConfig()
  input_fn_train = zh_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    input_files=[TRAIN_FILE],
    batch_size=hparams.batch_size,
    num_epochs=FLAGS.num_epochs)

  input_fn_eval = zh_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size,
    num_epochs=1)

  eval_metrics = zh_metrics.create_evaluation_metrics()

  eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)#喂数据


  estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])



if __name__ == "__main__":
  tf.app.run()
