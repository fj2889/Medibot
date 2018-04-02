import tensorflow as tf

def process_state(states):
    ''''
    输入状态或者每个时间点的状态，处理后砍成两半
    1.输入的是列表，说明是双向网络的输出
      ①检查是GRU还是LSTM的输出
    2.
    '''
    if isinstance(states, tuple):
        if isinstance(states[0], tf.nn.rnn_cell.LSTMStateTuple):
            states = tf.concat([states[0].h, states[1].h], 1)
            # encoding_context, encoding_utterance = tf.split(states, 2, 0)
        else:
            states = tf.concat([states[0], states[1]], 1)
        encoding_context, encoding_utterance = tf.split(states, 2, 0)
    else:
        if isinstance(states, tf.nn.rnn_cell.LSTMStateTuple):
            states = states.h
            # encoding_context, encoding_utterance = tf.split(states, 2, 0)
        encoding_context, encoding_utterance = tf.split(states, 2, 0)
    return encoding_context, encoding_utterance
#outputs,states=RNN(tf.nn.rnn_cell.GRUCell,hparams,context_embedded,context_len,utterance,utterance_len,False)
def RNN(
        RNN_Init,
        hparams,
        context_embedded,
        context_len,
        utterance_embedded,
        utterance_len,
        BiDirection=False):
    if not issubclass(RNN_Init,tf.nn.rnn_cell.RNNCell):
        raise ValueError('RNN_Init illegal!!!!!')
    if RNN_Init is tf.nn.rnn_cell.LSTMCell:
        cell_fw = RNN_Init(
            hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,  # 让 门层 也会接受细胞状态的输入
            state_is_tuple=True)
        if BiDirection:
            cell_bw = RNN_Init(
                num_units=hparams.rnn_dim,
                forget_bias=2.0,
                use_peepholes=True,  # 让 门层 也会接受细胞状态的输入
                state_is_tuple=True)
    elif RNN_Init is tf.nn.rnn_cell.BasicLSTMCell:
        cell_fw = RNN_Init(
            num_units=hparams.rnn_dim,
            forget_bias=2.0,
            state_is_tuple=True)
        if BiDirection:
            cell_bw = RNN_Init(
                num_units=hparams.rnn_dim,
                forget_bias=2.0,
                state_is_tuple=True)
    else:
        cell_fw = RNN_Init(num_units=hparams.rnn_dim)
        if BiDirection:
            cell_bw = RNN_Init(num_units=hparams.rnn_dim)

    if not BiDirection:
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell_fw,
            tf.concat([context_embedded, utterance_embedded], 0),
            sequence_length=tf.concat([context_len, utterance_len], 0),  # 指定长度就不需要padding了
            dtype=tf.float32)
    else:
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=tf.concat(
                [context_embedded, utterance_embedded], 0),
            sequence_length=tf.concat(
                [context_len, utterance_len], 0),
            dtype=tf.float32)
    return rnn_outputs,rnn_states


def RNN_MaxPooling(RNN_Init,
    hparams,
    context_embedded,
    context_len,
    utterance_embedded,
    utterance_len,
    BiDirection=False):
    outputs, states = RNN(RNN_Init, hparams, context_embedded, context_len, utterance_embedded, utterance_len, BiDirection)
    # OUTPUTS shape:[batch_size,sequence_length,dim]
    if isinstance(outputs, tuple):
        outputs = tf.concat([outputs[0], outputs[1]], axis=1)  # 双向GRU和LSTM就把前向网络和后向网络连接起来
    outputs = tf.expand_dims(outputs, -1)  # 增加通道维度

    '''
    池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    '''
    outputs = tf.nn.max_pool(outputs, ksize=[1, outputs.get_shape()[1], 1, 1],
                             strides=[1, 1, 1, 1], padding='VALID')  # [batch_size,sequence_length,dim,1]
    print('after maxpool {}'.format(outputs))
    outputs = tf.squeeze(outputs)
    print('after reshape {}'.format(outputs))
    # encoding_context, encoding_utterance = tf.split(outputs, 2, 0)
    encoding_context, encoding_utterance = process_state(outputs)
    print('after split {}'.format(outputs))
    return encoding_context, encoding_utterance
def RNN_CNN_MaxPooling(
    RNN_Init,
    hparams,
    context_embedded,
    context_len,
    utterance_embedded,
    utterance_len,
    BiDirection=False,
    filtersizes=[2,3,4,5],num_filters=30
):

  outputs, states = RNN(RNN_Init,hparams,context_embedded,context_len,utterance_embedded,utterance_len,BiDirection)
  #print('outputs shape {}'.format(outputs.get_shape()))
  if isinstance(outputs,tuple):
      outputs = tf.concat([outputs[0], outputs[1]], axis=1)  # 双向GRU和LSTM就把前向网络和后向网络连接起来
  outputs = tf.expand_dims(outputs, -1)  # 增加通道维度  shape:[batch_size,sequence_length,dim,1]
  print('outputs_expand shape {}'.format(outputs.get_shape()))
  outputs_context,outputs_utterance=tf.split(outputs,2,0)#分别为context和utterence
  print('outputs_context shape {}'.format(outputs_context.get_shape()))
  print('outputs_utterance shape {}'.format(outputs_utterance.get_shape()))
  pooled_context_list=[]
  pooled_utterance_list=[]
  for i,filter_size in enumerate(filtersizes):
      with tf.variable_scope('conv-maxpool-%s'%filter_size):
          filter_context_shape = [
                  filter_size,
                  int(outputs_context.get_shape()[2]),
                  1,
                  num_filters]
          W_context=tf.get_variable(
              name="W_context",
              initializer=tf.truncated_normal(filter_context_shape, stddev=0.1),
              dtype=tf.float32
              )
          b_context = tf.get_variable(name="b_context",initializer=tf.constant(0.1, shape=[num_filters]))
          conv_context=tf.nn.conv2d(
              outputs_context,
              W_context,
              strides=[1,1,1,1],
              padding='VALID',
              name='conv-context'
          )#(128, 159, 1, 30)
          print('conv_context shape {}'.format(conv_context.get_shape()))
          h_context=tf.nn.relu(tf.nn.bias_add(conv_context,b_context),name='h_context')
          pooled_context=tf.nn.max_pool(
              h_context,
              ksize=[1,h_context.get_shape()[1],1,1],
              strides=[1,1,1,1],
              padding='VALID',
              name='pool_context'
          )#(128, 1, 1, 30)
          print('pool_context shape {}'.format(pooled_context.get_shape()))
          pooled_context_list.append(pooled_context)

          filter_utterance_shape = [
              filter_size,
              int(outputs_utterance.get_shape()[2]),
              1,
              num_filters
          ]
          W_utterance = tf.get_variable(
              name="W_utterance",
              initializer=tf.truncated_normal(filter_utterance_shape, stddev=0.1)
              )
          b_utterance = tf.get_variable(name="b_utterance",initializer=tf.constant(0.1, shape=[num_filters]) )
          conv_utterance = tf.nn.conv2d(
              outputs_utterance,
              W_utterance,
              strides=[1, 1, 1, 1],
              padding='VALID',
              name='conv-utterance'
          )
          print('conv_utterance shape {}'.format(conv_utterance.get_shape()))
          h_utterance = tf.nn.relu(tf.nn.bias_add(conv_utterance, b_utterance), name='h_utterance')
          pooled_utterance = tf.nn.max_pool(
              h_utterance,
              ksize=[1, h_utterance.get_shape()[1], 1, 1],
              strides=[1,1,1,1],
              padding='VALID',
              name='pool_utterance'
          )#batch_size,1,1,num_filters
          print('pooled_utterance shape {}'.format(pooled_utterance.get_shape()))
          pooled_utterance_list.append(pooled_utterance)
  encoding_context=tf.concat(pooled_context_list,3)
  encoding_utterance=tf.concat(pooled_utterance_list,3)
  print('concact_context shape {}'.format(encoding_context.get_shape()))
  encoding_context=tf.reshape(encoding_context,shape=[-1,num_filters*len(filtersizes)])
  encoding_utterance=tf.reshape(encoding_utterance,shape=[-1,num_filters*len(filtersizes)])
  print('reshape_context shape {}'.format(encoding_context.get_shape()))
  return encoding_context,encoding_utterance
def RNN_Attention(RNN_Init,
                hparams,
                context_embedded,
                context_len,
                utterance_embedded,
                utterance_len,
                BiDirection=False):

  outputs, states = RNN(RNN_Init, hparams, context_embedded, context_len, utterance_embedded, utterance_len,
                   BiDirection)# OUTPUTS shape:[batch_size,sequence_length,dim]
  if isinstance(outputs, tuple):
      outputs = tf.concat([outputs[0], outputs[1]], axis=1)  # 双向GRU和LSTM就把前向网络和后向网络连接起来
  outputs_context, outputs_utterance = tf.split(outputs, 2, 0)  # 分别为context和utterence
  outputs_context=tf.expand_dims(outputs_context,[-1])
  #得到问题的embedded 向量
  outputs_context = tf.nn.max_pool(outputs_context,
                                   ksize=[1, outputs_context.get_shape()[1], 1, 1],
                                   strides=[1, 1, 1, 1],
                                   padding='VALID')  # [batch_size,1,dim,1]
  outputs_context=tf.squeeze(outputs_context,axis=[3])#删除通道维度
  with tf.variable_scope('attention'):
      Wam=tf.get_variable(name='Wam',
                          shape=[outputs_utterance.get_shape()[2],1],
                          initializer=tf.truncated_normal_initializer(stddev=0.1))#wam*ht->batch,1
      Wqm=tf.get_variable(name='Wqm',
                          shape=[outputs_context.get_shape()[2],1],
                          initializer=tf.truncated_normal_initializer(stddev=0.1))#oq*wqm->batch,1
      Wam_stack_batch=tf.reshape(
          tf.tile(Wam,[int(outputs_utterance.get_shape()[0]),1]),
          shape=[
              -1,
              int(Wam.get_shape()[0]),
              int(Wam.get_shape()[1])
          ])
      Wqm_stack_batch = tf.reshape(
          tf.tile(Wqm, [int(outputs_context.get_shape()[0]), 1]),
          shape=[
              -1,
              int(Wqm.get_shape()[0]),
              int(Wqm.get_shape()[1])
          ])
      utterance_Mat=tf.matmul(outputs_utterance, Wam_stack_batch)#(128, 160, 1)
      print("answer_mat:{}".format(utterance_Mat.get_shape()))
      utterance_Mat=tf.squeeze(utterance_Mat,[2])#(128, 160)
      print("answer_mat:{}".format(utterance_Mat.get_shape()))
      context_Mat=tf.matmul(outputs_context, Wqm_stack_batch)
      print("context_mat:{}".format(context_Mat.get_shape()))
      context_Mat=tf.squeeze(context_Mat,[2])
      print("context_mat:{}".format(context_Mat.get_shape()))
      maq=tf.nn.tanh(utterance_Mat+context_Mat)
      print("maq:{}".format(maq.get_shape()))

      Wms=tf.get_variable(name='Wms',
                          shape=[outputs_utterance.get_shape()[1],outputs_utterance.get_shape()[1]],
                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                          )
      saq=tf.nn.softmax(tf.matmul(maq,Wms))#saq:(128, 160)
      print("saq:{}".format(saq.get_shape()))
      #element-wise
      outputs_utterance=tf.matmul(tf.reshape(saq,[-1,1,int(saq.get_shape()[1])]),outputs_utterance)
      outputs_utterance=tf.expand_dims(outputs_utterance,[-1])
      outputs_utterance = tf.nn.max_pool(outputs_utterance,
                                       ksize=[1, outputs_utterance.get_shape()[1], 1, 1],
                                       strides=[1, 1, 1, 1],
                                       padding='VALID')  # [batch_size,1,dim,1]
      outputs_utterance=tf.squeeze(outputs_utterance)
      outputs_context=tf.squeeze(outputs_context)
      return outputs_context,outputs_utterance