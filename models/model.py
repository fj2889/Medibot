import tensorflow as tf
from models import helpers
FLAGS = tf.flags.FLAGS

def process_state(states):
    ''''
    输入状态或者每个时间点的状态，处理后砍成两半
    1.输入的是列表，说明是双向网络的输出
      ①检查是GRU还是LSTM的输出
    2.
    '''
    if isinstance(states,tf.nn.rnn_cell.LSTMStateTuple):
        #单向LSTM的输出，是一个namedtuple，我们需要其中的h部分
        states = states.h
    elif isinstance(states, tuple):
        if(len(states)!=2):
            raise ValueError('states illegal')
        m=isinstance(states[0], tf.nn.rnn_cell.LSTMStateTuple)
        n=isinstance(states[1],tf.nn.rnn_cell.LSTMStateTuple)
        #沿着最后一个axis拼接，默认最后一个维度代表feature，这样不强制states[0]是二维的数据，也可以是三维的outputs
        if(m and n):
            states = tf.concat([states[0].h, states[1].h], -1)
        elif(m and not n):
            states = tf.concat([states[0].h, states[1]], -1)#m=true n=false
        elif(not m and n):
            states = tf.concat([states[0], states[1].h], -1)
        else:
            states = tf.concat([states[0], states[1]], -1)
    encoding_context, encoding_utterance = tf.split(states, 2, 0)
    return encoding_context, encoding_utterance


def RNN(
        RNN_Init,
        hparams,
        context_embedded,
        context_len,
        utterance_embedded,
        utterance_len,
        is_bidirection=False):
    if not issubclass(RNN_Init, tf.nn.rnn_cell.RNNCell):
        raise ValueError('RNN_Init illegal!!!!!')

    if RNN_Init is tf.nn.rnn_cell.LSTMCell:
        cell_fw = RNN_Init(
            hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,  # 让 门层 也会接受细胞状态的输入
            state_is_tuple=True)
        if is_bidirection:
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
        if is_bidirection:
            cell_bw = RNN_Init(
                num_units=hparams.rnn_dim,
                forget_bias=2.0,
                state_is_tuple=True)
    else:
        cell_fw = RNN_Init(num_units=hparams.rnn_dim)
        if is_bidirection:
            cell_bw = RNN_Init(num_units=hparams.rnn_dim)

    if not is_bidirection:
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell_fw,
            tf.concat([context_embedded, utterance_embedded], 0),
            sequence_length=tf.concat(
                [context_len, utterance_len], 0),  # 指定长度就不需要padding了
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
    return rnn_outputs, rnn_states

def RNN_MaxPooling(RNN_Init,
                   hparams,
                   context_embedded,
                   context_len,
                   utterance_embedded,
                   utterance_len,
                   is_bidirection=False):
    outputs, states = RNN(RNN_Init, hparams, context_embedded,
                          context_len, utterance_embedded, utterance_len, is_bidirection)
    # OUTPUTS shape:[batch_size,sequence_length,dim]
    if isinstance(outputs, tuple):
        # 双向GRU和LSTM就把前向网络和后向网络连接起来
        outputs = tf.concat([outputs[0], outputs[1]], axis=-1)#按最后一列feature拼接!!
    print(outputs.get_shape())
    outputs = tf.expand_dims(outputs, -1)  # 增加通道维度

    '''
    池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    '''
    outputs = tf.nn.max_pool(outputs, ksize=[1, outputs.get_shape()[1], 1, 1],
                             strides=[1, 1, 1, 1], padding='VALID')  # [batch_size,sequence_length,dim,1]
    print('after maxpool {}'.format(outputs))
    outputs = tf.squeeze(outputs,axis=[1,3])
    print('after reshape {}'.format(outputs))

    encoding_context, encoding_utterance = tf.split(outputs, 2, 0)
    #encoding_context, encoding_utterance = process_state(outputs)
    return encoding_context, encoding_utterance


def RNN_CNN_MaxPooling(
        RNN_Init,
        hparams,
        context_embedded,
        context_len,
        utterance_embedded,
        utterance_len,
        BiDirection=False,
        filtersizes=[2, 3, 4, 5],
        num_filters=30
):
    outputs, states = RNN(RNN_Init, hparams, context_embedded,
                          context_len, utterance_embedded, utterance_len, BiDirection)
    # print('outputs shape {}'.format(outputs.get_shape()))
    if isinstance(outputs, tuple):
        # 双向GRU和LSTM就把前向网络和后向网络连接起来
        outputs = tf.concat([outputs[0], outputs[1]], axis=-1)
    print(outputs.get_shape())
    # 增加通道维度  shape:[batch_size,sequence_length,dim,1]
    outputs = tf.expand_dims(outputs, -1)
    print('outputs_expand shape {}'.format(outputs.get_shape()))
    outputs_context, outputs_utterance = tf.split(
        outputs, 2, 0)  # 分别为context和utterence
    print('outputs_context shape {}'.format(outputs_context.get_shape()))
    print('outputs_utterance shape {}'.format(outputs_utterance.get_shape()))

    def map_filter(filter_size):
        with tf.variable_scope('conv-maxpool-%s' % filter_size):
            filter_context_shape = [
                filter_size,
                int(outputs_context.get_shape()[2]),
                1,
                num_filters]
            W_context = tf.get_variable(
                name="W_context",
                initializer=tf.truncated_normal(
                    filter_context_shape, stddev=0.1),
                dtype=tf.float32
            )
            b_context = tf.get_variable(
                name="b_context", initializer=tf.constant(0.1, shape=[num_filters]))
            conv_context = tf.nn.conv2d(
                outputs_context,
                W_context,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv-context'
            )  # (128, 159, 1, 30)
            print('conv_context shape {}'.format(conv_context.get_shape()))
            h_context = tf.nn.relu(tf.nn.bias_add(
                conv_context, b_context), name='h_context')
            pooled_context = tf.nn.max_pool(
                h_context,
                ksize=[1, h_context.get_shape()[1], 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool_context'
            )  # (128, 1, 1, 30)
            print('pool_context shape {}'.format(pooled_context.get_shape()))
            # pooled_context_list.append(pooled_context)

            filter_utterance_shape = [
                filter_size,
                int(outputs_utterance.get_shape()[2]),
                1,
                num_filters
            ]
            W_utterance = tf.get_variable(
                name="W_utterance",
                initializer=tf.truncated_normal(
                    filter_utterance_shape, stddev=0.1)
            )
            b_utterance = tf.get_variable(
                name="b_utterance", initializer=tf.constant(0.1, shape=[num_filters]))
            conv_utterance = tf.nn.conv2d(
                outputs_utterance,
                W_utterance,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv-utterance'
            )
            print('conv_utterance shape {}'.format(conv_utterance.get_shape()))
            h_utterance = tf.nn.relu(tf.nn.bias_add(
                conv_utterance, b_utterance), name='h_utterance')
            pooled_utterance = tf.nn.max_pool(
                h_utterance,
                ksize=[1, h_utterance.get_shape()[1], 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool_utterance'
            )  # batch_size,1,1,num_filters
            print('pooled_utterance shape {}'.format(
                pooled_utterance.get_shape()))
            # pooled_utterance_list.append(pooled_utterance)
        return [pooled_context, pooled_utterance]

    _list = list(map(map_filter, filtersizes))

    pooled_context_list = [x[0] for x in _list]
    pooled_utterance_list = [x[1] for x in _list]
    encoding_context = tf.concat(pooled_context_list, 3)
    encoding_utterance = tf.concat(pooled_utterance_list, 3)
    print('concact_context shape {}'.format(encoding_context.get_shape()))
    encoding_context = tf.reshape(
        encoding_context, shape=[-1, num_filters * len(filtersizes)])
    encoding_utterance = tf.reshape(
        encoding_utterance, shape=[-1, num_filters * len(filtersizes)])
    print('reshape_context shape {}'.format(encoding_context.get_shape()))
    print('reshape_utterence shape {}'.format(encoding_utterance.get_shape()))
    return encoding_context, encoding_utterance


def RNN_Attention(RNN_Init,
                  hparams,
                  context_embedded,
                  context_len,
                  utterance_embedded,
                  utterance_len,
                  BiDirection=False):
    outputs, states = RNN(RNN_Init, hparams, context_embedded, context_len, utterance_embedded, utterance_len,
                          BiDirection)  # OUTPUTS shape:[batch_size,sequence_length,dim]
    if isinstance(outputs, tuple):
        # 双向GRU和LSTM就把前向网络和后向网络连接起来
        outputs = tf.concat([outputs[0], outputs[1]], axis=-1)
    outputs_context, outputs_utterance = tf.split(
        outputs, 2, 0)  # 分别为context和utterence
    outputs_context = tf.expand_dims(outputs_context, [-1])
    # 得到问题的embedded 向量
    outputs_context = tf.nn.max_pool(outputs_context,
                                     ksize=[1, outputs_context.get_shape()[
                                         1], 1, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='VALID')  # [batch_size,1,dim,1]
    outputs_context = tf.squeeze(outputs_context, axis=[3])  # 删除通道维度
    with tf.variable_scope('attention'):
        Wam = tf.get_variable(name='Wam',
                              shape=[outputs_utterance.get_shape()[2], 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))  # wam*ht->batch,1
        Wqm = tf.get_variable(name='Wqm',
                              shape=[outputs_context.get_shape()[2], 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))  # oq*wqm->batch,1
        Wam_stack_batch = tf.reshape(
            tf.tile(Wam, [int(outputs_utterance.get_shape()[0]), 1]),
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
        utterance_Mat = tf.matmul(
            outputs_utterance, Wam_stack_batch)  # (128, 160, 1)
        print("answer_mat:{}".format(utterance_Mat.get_shape()))
        utterance_Mat = tf.squeeze(utterance_Mat, [2])  # (128, 160)
        print("answer_mat:{}".format(utterance_Mat.get_shape()))
        context_Mat = tf.matmul(outputs_context, Wqm_stack_batch)
        print("context_mat:{}".format(context_Mat.get_shape()))
        context_Mat = tf.squeeze(context_Mat, [2])
        print("context_mat:{}".format(context_Mat.get_shape()))
        maq = tf.nn.tanh(utterance_Mat + context_Mat)
        print("maq:{}".format(maq.get_shape()))

        Wms = tf.get_variable(name='Wms',
                              shape=[outputs_utterance.get_shape(
                              )[1], outputs_utterance.get_shape()[1]],
                              initializer=tf.truncated_normal_initializer(
                                  stddev=0.1)
                              )
        saq = tf.nn.softmax(tf.matmul(maq, Wms))  # saq:(128, 160)
        print("saq:{}".format(saq.get_shape()))
        # element-wise
        outputs_utterance = tf.matmul(tf.reshape(
            saq, [-1, 1, int(saq.get_shape()[1])]), outputs_utterance)
        outputs_utterance = tf.expand_dims(outputs_utterance, [-1])
        outputs_utterance = tf.nn.max_pool(outputs_utterance,
                                           ksize=[1, outputs_utterance.get_shape()[
                                               1], 1, 1],
                                           strides=[1, 1, 1, 1],
                                           padding='VALID')  # [batch_size,1,dim,1]
        outputs_utterance = tf.squeeze(outputs_utterance)
        outputs_context = tf.squeeze(outputs_context)
        return outputs_context, outputs_utterance




def get_embeddings(hparams):
    if hparams.word2vec_path and hparams.vocab_path:
        tf.logging.info("Loading Glove embeddings...")

        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        # glove_vectors ndarray
        # glove_dict dict
        glove_vectors, glove_dict = helpers.load_glove_vectors(
            hparams.word2vec_path, vocab=set(vocab_array))

        initializer = helpers.build_initial_embedding_matrix(
            vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
        return tf.get_variable(
            "word_embeddings",
            initializer=initializer)
    else:
        tf.logging.info(
            "No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

        return tf.get_variable(
            "word_embeddings",
            shape=[hparams.vocab_size, hparams.embedding_dim],
            initializer=initializer)


def dual_encoder_model(
        hparams,
        mode,
        context,
        context_len,
        utterance,
        utterance_len,
        targets,
        model_fun,
        RNNInit=tf.nn.rnn_cell.LSTMCell,
        is_bidirection=False):  # 默认单向

    print("input context shape{}".format(context.get_shape()))
    print("input utterance shape{}".format(utterance.get_shape()))
    print("input context_len shape{}".format(context_len.get_shape()))
    print("input utterance_len shape{}".format(utterance_len.get_shape()))
    print("error targets shape{}:".format(targets.get_shape()))
    # Initialize embedidngs randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(hparams)
    print('context shape {}'.format(context))  # 顺便显示shape
    # Embed the context and the utterance
    context_embedded = tf.nn.embedding_lookup(  # 三维？
        embeddings_W, context, name="embed_context")
    utterance_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterance, name="embed_utterance")

    print('context_embedded shape {}'.format(context_embedded.get_shape()))
    print('utterence_embedded shape {}'.format(utterance_embedded.get_shape()))

    if model_fun is RNN:
        outputs, states = RNN(RNNInit,
                                    hparams,
                                    context_embedded,
                                    context_len,
                                    utterance_embedded,
                                    utterance_len,
                                    is_bidirection)
        encoding_context, encoding_utterance = process_state(states)
    elif model_fun in set([RNN_MaxPooling, RNN_CNN_MaxPooling, RNN_CNN_MaxPooling, RNN_Attention]):
        encoding_context, encoding_utterance = model_fun(RNNInit,
                                                         hparams,
                                                         context_embedded,
                                                         context_len,
                                                         utterance_embedded,
                                                         utterance_len,
                                                         is_bidirection)
    else:
        raise ValueError('model_fun illegal!!!!!!')
    print('encoding_context type {}'.format(type(encoding_context)))
    print('encoding_context shape {}'.format(encoding_context.get_shape()))
    print('encoding_utterence shape {}'.format(encoding_utterance.get_shape()))
    with tf.variable_scope("prediction"):
        _shape = encoding_context.get_shape()[1].value
        print('M_shape {}'.format(encoding_context.get_shape()[1].value))
        M = tf.get_variable("M",
                            shape=[_shape,_shape],
                            initializer=tf.truncated_normal_initializer())

        # "Predict" a  response: c * M
        generated_response = tf.matmul(encoding_context, M)
        print('c*M {}'.format((generated_response.get_shape())))
        generated_response = tf.expand_dims(
            generated_response, 2)  # 增加了第三个维度？？
        # 增加维度是为了让每个batch分开做乘积，也就是乘积只做用在后面两个维度
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        print('expand dims generated response {}'.format(generated_response.get_shape()))
        print('expand dims encoding utterence {}'.format(encoding_utterance.get_shape()))
        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response,
                           encoding_utterance, True)  # 维度匹配
        print('logits shape {}'.format(logits))
        logits = tf.squeeze(logits, [2])  # 删除第三个维度?
        print('squeeze logits shape {}'.format(logits.get_shape()))
        # Apply sigmoid to convert logits to probabilities
        probs = tf.sigmoid(logits)#没有问题，是sigmoid不是softmax

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None


        print("error logits shape{}:".format(logits.get_shape()))
        print("error targets shape{}:".format(targets.get_shape()))
        # Calculate the binary cross-entropy loss
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss
