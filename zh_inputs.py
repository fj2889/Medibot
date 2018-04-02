import tensorflow as tf

TEXT_FEATURE_SIZE = 160

def get_feature_columns(mode):
  feature_columns = []
  #输进去一个问题，问题长度和答案长度
  feature_columns.append(tf.contrib.layers.real_valued_column(
    column_name="context", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))#每一个问题和答案最大长度
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="context_len", dimension=1, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="utterance", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="utterance_len", dimension=1, dtype=tf.int64))

  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    # During training we have a label feature
    feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="label", dimension=1, dtype=tf.int64))

  if mode == tf.contrib.learn.ModeKeys.EVAL:
    # During evaluation we have distractors
    for i in range(9):
      feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="distractor_{}".format(i), dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))#输入9个错误答案
      feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))

  return set(feature_columns)


'''
包含将特征列名称映射到包含相应特征数据的`Tensor`（或`SparseTensor`）的键/值对的字典。
包含您的标签（目标）值的`Tensor`：你的模型的值的目的是用于预测。
'''
def create_input_fn(mode, input_files, batch_size, num_epochs):
  def input_fn():
    features = tf.contrib.layers.create_feature_spec_for_parsing(        #???
        get_feature_columns(mode))
    #从输入文件创建batch
    ''''
    THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Use tf.data
    Given file pattern (or list of files), will setup a queue for file names, read Example proto using provided 
    reader, use batch queue to create batches of examples of size batch_size and parse example given features specification.
    '''
    feature_map = tf.contrib.learn.io.read_batch_features(
        file_pattern=input_files,
        batch_size=batch_size,
        features=features,
        reader=tf.TFRecordReader,
        randomize_input=True,
        num_epochs=num_epochs,
        queue_capacity=200000 + batch_size * 10,           #洗牌 shuffle
        name="read_batch_features_{}".format(mode))

    # This is an ugly hack because of a current bug in tf.learn
    # During evaluation TF tries to restore the epoch variable which isn't defined during training
    # So we define the variable manually here
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      tf.get_variable(
        "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
        initializer=tf.constant(0, dtype=tf.int64))

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      target = feature_map.pop("label")
    else:
      # In evaluation we have 10 classes (utterances).
      # The first one (index 0) is always the correct one
      target = tf.zeros([batch_size, 1], dtype=tf.int64)
    return feature_map, target
  return input_fn
