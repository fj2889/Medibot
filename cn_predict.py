import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import cn_model
import cn_hparams
import cn_metrics
import cn_inputs
from models.model import dual_encoder_model
from models import model
from make_data import Dataset
from models.helpers import load_vocab
import pickle
from operator import itemgetter, attrgetter

tf.flags.DEFINE_string("model_dir", "./runs/RNN_CNN_MaxPooling", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
with open("data/dataset.pkl","rb") as file:
    dataset = pickle.load(file)

INPUT_CONTEXT = dataset.raw_data[13]["question"]
# POTENTIAL_RESPONSES = [dataset.raw_data[i]["answer"] for i in range(len(dataset.raw_data))]
POTENTIAL_RESPONSES = [dataset.raw_data[i]["answer"] for i in range(20)]

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = cn_hparams.create_hparams()
  model_fn = cn_model.create_model_fn(
                hparams,
                model_impl=dual_encoder_model,
                model_fun=model.RNN_CNN_MaxPooling,
                RNNInit=tf.nn.rnn_cell.LSTMCell,
                is_bidirection=True)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  # estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  print("Context: {}".format(INPUT_CONTEXT))
  probs = []
  for num, r in enumerate(POTENTIAL_RESPONSES):
    probs.append({num: estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))})
  sorted(probs, key=itemgetter(1))
  for i in probs:
      print(i)