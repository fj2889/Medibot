'''import tensorflow as tf
import functools
from tensorflow.contrib.learn import MetricSpec
from tensorflow.contrib.metrics import streaming_sparse_recall_at_k
def create_evaluation_metrics():            #??
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k,
            k=k))
    return eval_metrics'''
import tensorflow as tf
import functools
from tensorflow.contrib.learn import MetricSpec
from tensorflow.contrib.metrics import streaming_sparse_recall_at_k

def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(streaming_sparse_recall_at_k,k=k))
    return eval_metrics
    #streaming_sparse_recall_at_k
# tf.contrib.metrics.streaming_sparse_recall_at_k(
#     predictions,            [0.34, 0.11, 0.22, 0.45, 0.01, 0.02, 0.03, 0.08, 0.33, 0.11]
#     labels,
#     k,
#     class_id=None,
#     weights=None,
#     metrics_collections=None,
#     updates_collections=None,
#     name=None
# )