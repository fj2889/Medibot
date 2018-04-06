
import tensorflow as tf
import functools
from tensorflow.contrib.learn import MetricSpec
from tensorflow.contrib.metrics import streaming_sparse_recall_at_k

def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(streaming_sparse_recall_at_k,k=k))
    return eval_metrics
