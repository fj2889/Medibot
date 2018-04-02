import os
import time
import itertools
import tensorflow as tf
import zh_model
import zh_hparams
import zh_metrics
import zh_inputs
from models.dual_encoder import dual_encoder_model

# 避免警告
os.environ['tf CPP MIN LOG level']= '2'

tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 100, "Evaluate after this many train steps")
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

  model_fn = zh_model.create_model_fn(
    hparams,
    model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

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
        # TODO
        # every_n_steps
        # 评估监视器
        # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        #     test_set.data,
        #     test_set.target,
        #     every_n_steps=50,
        #     metrics=validation_metrics)
        metrics=eval_metrics)

  estimator.fit(input_fn=input_fn_train,
                steps=None,
                monitors=[eval_monitor])

if __name__ == "__main__":
  tf.app.run()
