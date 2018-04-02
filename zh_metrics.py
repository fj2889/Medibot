import tensorflow as tf
import functools
from tensorflow.contrib.metrics import streaming_sparse_recall_at_k
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.python.ops import math_ops


def create_evaluation_metrics():  # ??
    eval_metrics = {}
    for _k in [1,2,5,10]:
        eval_metrics["recall_at_%d" % _k] = MetricSpec(metric_fn=functools.partial(
            streaming_sparse_recall_at_k,
            k=_k))
    # print(eval_metrics['recall_at_1'].__dict__)
    return eval_metrics

    # validation_metrics = {
    #     "accuracy":
    #         tf.contrib.learn.MetricSpec(
    #             metric_fn=my_accuracy)
    #     # "precision":
    #     #     tf.contrib.learn.MetricSpec(
    #     #         metric_fn=my_accuracy,
    #     #         prediction_key=tf.contrib.learn.PredictionKey.
    #     #         CLASSES),
    #     # "recall":
    #     #     tf.contrib.learn.MetricSpec(
    #     #         metric_fn=my_accuracy,
    #     #         prediction_key=tf.contrib.learn.PredictionKey.
    #     #         CLASSES)
    # }
    # return validation_metrics

# 您可以自定义metric函数，它们必须采用predictions和labels张量作为参数（weights也可以成为可选参数）。该函数必须以两种格式之一返回metric值：
#

# 单一张量
# 一对(value_op, update_op)操作，value_op返回metric值，update_op执行相应的操作来更新模型内部状态。
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


def my_accuracy(predictions,
                labels):
    """Calculates how often `predictions` matches `labels`.

    Returns:
      accuracy: A `Tensor` representing the accuracy, the value of `total` divided
        by `count`.
      update_op: An operation that increments the `total` and `count` variables
        appropriately and whose value matches `accuracy`.

    Raises:
      ValueError: If `predictions` and `labels` have mismatched shapes, or if
        `weights` is not `None` and its shape doesn't match `predictions`, or if
        either `metrics_collections` or `updates_collections` are not a list or
        tuple.
    """
    # if labels.dtype != predictions.dtype:
    #     predictions = math_ops.cast(predictions, labels.dtype)
    predictions=predictions[0:16]
    if labels.dtype != predictions.dtype:
        # labels = math_ops.cast(labels,predictions.dtype )
        predictions = math_ops.cast(predictions,labels.dtype )
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))



    return tf.metrics.mean(is_correct,  name='accuracy')
