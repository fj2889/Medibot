import tensorflow as tf
from models import helpers
FLAGS = tf.flags.FLAGS



class Model_Parameter(object):
    def __init__(self,RNN_Init,
        hparams,
        context_embedded,
        context_len,
        utterance_embedded,
        utterance_len,
        is_bidirection=False,
        input_keep_prob=1.0, output_keep_prob=1.0):

        self.RNN_Init=RNN_Init
        self.hparams=hparams
        self.context_embedded=context_embedded
        self.context_len=context_len
        self.utterance_embedded=utterance_embedded
        self.utterance_len=utterance_len
        self.is_bidirection = is_bidirection
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob


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


def RNN(model_parameter):
    if not isinstance(model_parameter,Model_Parameter):
        raise ValueError('model parameter illegal!')

    if model_parameter.RNN_Init is tf.nn.rnn_cell.LSTMCell:
        cell_fw = model_parameter.RNN_Init(
            model_parameter.hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,  # 让 门层 也会接受细胞状态的输入
            state_is_tuple=True)
        if model_parameter.is_bidirection:
            cell_bw = model_parameter.RNN_Init(
                num_units=model_parameter.hparams.rnn_dim,
                forget_bias=2.0,
                use_peepholes=True,  # 让 门层 也会接受细胞状态的输入
                state_is_tuple=True)
    elif model_parameter.RNN_Init is tf.nn.rnn_cell.BasicLSTMCell:
        cell_fw = model_parameter.RNN_Init(
            num_units=model_parameter.hparams.rnn_dim,
            forget_bias=2.0,
            state_is_tuple=True)
        if model_parameter.is_bidirection:
            cell_bw = model_parameter.RNN_Init(
                num_units=model_parameter.hparams.rnn_dim,
                forget_bias=2.0,
                state_is_tuple=True)
    else:
        cell_fw = model_parameter.RNN_Init(num_units=model_parameter.hparams.rnn_dim)
        if model_parameter.is_bidirection:
            cell_bw = model_parameter.RNN_Init(num_units=model_parameter.hparams.rnn_dim)


    if not model_parameter.is_bidirection:
        cell_fw=tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                              input_keep_prob=model_parameter.input_keep_prob,
                                              output_keep_prob=model_parameter.output_keep_prob)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell_fw,
            tf.concat([model_parameter.context_embedded, model_parameter.utterance_embedded], 0),
            sequence_length=tf.concat(
                [model_parameter.context_len, model_parameter.utterance_len], 0),  # 指定长度就不需要padding了
            dtype=tf.float32)
    else:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                                input_keep_prob=model_parameter.input_keep_prob,
                                                output_keep_prob=model_parameter.output_keep_prob)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,
                                                input_keep_prob=model_parameter.input_keep_prob,
                                                output_keep_prob=model_parameter.output_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=tf.concat(
                [model_parameter.context_embedded, model_parameter.utterance_embedded], 0),
            sequence_length=tf.concat(
                [model_parameter.context_len, model_parameter.utterance_len], 0),
            dtype=tf.float32)
    return rnn_outputs, rnn_states

def RNN_MaxPooling(model_parameter):
    outputs, states = RNN(model_parameter)
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

def __conv_pooling(filter_size,num_filters,name,input):
    input_shape=input.get_shape().as_list()
    if len(input_shape)!=4:
        raise ValueError("INPUT illegal!")

    sequence_length=int(input_shape[1])
    dim=int(input_shape[2])

    with tf.variable_scope((name+'-conv-maxpool-%s') % filter_size):
        conv_shape=[filter_size,dim,1,num_filters]

        W=tf.get_variable(name='W',initializer=tf.truncated_normal(conv_shape,stddev=0.1),dtype=tf.float32)

        b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[num_filters]))
        conv = tf.nn.conv2d(
            input,
            W,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name=(name+'-conv-%s') % filter_size
        )  # (128, 159, 1, 30)
        print('conv_context shape {}'.format(conv.get_shape()))
        conv= tf.nn.leaky_relu(tf.nn.bias_add(conv, b), name=(name+'-conv-maxpool-%s') % filter_size)#!!!!!

        pool = tf.nn.max_pool(
            conv,
            ksize=[1, sequence_length-filter_size+1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name=(name+'-pool-%s') % filter_size
        )  # (128, 1, 1, 30)
        print('pool_context shape {}'.format(pool.get_shape()))
    return pool



import functools
def RNN_CNN_MaxPooling(model_parameter,
        filtersizes=[2,3,4,5],
        num_filters=100
):
    outputs, states = RNN(model_parameter)
    # print('outputs shape {}'.format(outputs.get_shape()))
    if isinstance(outputs, tuple):
        # 双向GRU和LSTM就把前向网络和后向网络连接起来
        outputs = tf.concat([outputs[0], outputs[1]], axis=-1)
    print(outputs.get_shape())
    # 增加通道维度  shape:[batch_size,sequence_length,dim,1]

    outputs_context, outputs_utterance = tf.split(
        outputs, 2, 0)  # 分别为context和utterence
    print('outputs_context shape {}'.format(outputs_context.get_shape()))
    print('outputs_utterance shape {}'.format(outputs_utterance.get_shape()))

    # outputs_context=model_parameter.context_embedded
    # outputs_utterance=model_parameter.utterance_embedded

    outputs_context=tf.expand_dims(outputs_context,[-1])

    __context_conv_fun=functools.partial(__conv_pooling,name='context',
                                         input=outputs_context,num_filters=num_filters)
    pooled_context_list = list(map(__context_conv_fun, filtersizes))
    encoding_context = tf.concat(pooled_context_list, 3)
    encoding_context = tf.reshape(
        encoding_context, shape=[-1, num_filters * len(filtersizes)])

    #outputs_utterance=__attention(ha=outputs_utterance,oq=encoding_context)

    outputs_utterance = tf.expand_dims(outputs_utterance, [-1])
    __utterance_conv_fun = functools.partial(__conv_pooling, name='utterance',
                                           input=outputs_utterance, num_filters=num_filters)
    pooled_utterance_list = list(map(__utterance_conv_fun,filtersizes))
    encoding_utterance = tf.concat(pooled_utterance_list, 3)
    encoding_utterance = tf.reshape(
        encoding_utterance, shape=[-1, num_filters * len(filtersizes)])

    print('reshape_context shape {}'.format(encoding_context.get_shape()))
    print('reshape_utterence shape {}'.format(encoding_utterance.get_shape()))
    return encoding_context, encoding_utterance


def RNN_Attention(model_parameter):
    outputs, states = RNN(model_parameter)  # OUTPUTS shape:[batch_size,sequence_length,dim]
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
    outputs_context = tf.squeeze(outputs_context, axis=[1,3])  # 删除通道维度

    outputs_utterance=__attention(outputs_utterance,outputs_context)

    outputs_utterance=tf.expand_dims(outputs_utterance,[-1])
    outputs_utterance = tf.nn.max_pool(outputs_utterance,
                                       ksize=[1, outputs_utterance.get_shape()[
                                           1], 1, 1],
                                       strides=[1, 1, 1, 1],
                                       padding='VALID')  # [batch_size,1,dim,1]

    outputs_utterance = tf.squeeze(outputs_utterance,axis=[1,3])
    return outputs_context, outputs_utterance


def __attention(ha,oq):
    '''

    :param ha: shape->(batch,t,dim)
    :param oq:shape->(batch,dim)
    :return:
    '''
    sequence_length = int(ha.get_shape()[1])
    dim_ha = int(ha.get_shape()[-1])
    dim_oq=int(oq.get_shape()[-1])

    with tf.variable_scope('attention'):
        Wam = tf.get_variable(name='Wam',
                              shape=[dim_ha, 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))  # wam*ht->batch,1
        Wqm = tf.get_variable(name='Wqm',
                              shape=[dim_oq, 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))  # oq*wqm->batch,1
        oq_Mat = tf.matmul(oq, Wqm)

        ha=tf.reshape(ha, [-1, dim_ha])
        ha_Mat=tf.matmul(ha,Wam)
        ha_Mat=tf.reshape(ha_Mat,[-1,sequence_length])

        b = tf.get_variable(name="attention_bias", initializer=tf.constant(0.1))
        maq=tf.nn.leaky_relu(ha_Mat+oq_Mat+b)#!!!!!

        Wms = tf.get_variable(name='Wms',
                              shape=[sequence_length, sequence_length],
                              initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))

        saq = tf.nn.softmax(tf.matmul(maq, Wms))  # saq:(128, 160)


        saq = tf.reshape(saq, shape=[-1, 1])
        ha = ha * saq

        ha=tf.reshape(ha,[-1,sequence_length,dim_ha])
    return ha


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
        RNN_Init,
        is_bidirection=False,
        input_keep_prob=1.0,
        output_keep_prob=1.0):  # 默认单向


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

    model_parameter=Model_Parameter(RNN_Init=RNN_Init,
                                    hparams=hparams,
                                    context_embedded=context_embedded,
                                    context_len=context_len,
                                    utterance_embedded=utterance_embedded,
                                    utterance_len=utterance_len,
                                    is_bidirection=is_bidirection,
                                    input_keep_prob=input_keep_prob,
                                    output_keep_prob=output_keep_prob)
    print(type(model_parameter))
    if model_fun is RNN:
        outputs, states = RNN(model_parameter)
        encoding_context, encoding_utterance = process_state(states)
    elif model_fun in set([RNN_MaxPooling, RNN_CNN_MaxPooling, RNN_CNN_MaxPooling, RNN_Attention])\
            or model_fun.func is RNN_CNN_MaxPooling:
        encoding_context, encoding_utterance = model_fun(model_parameter)
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

        #计算范数
        generated_response_norm=tf.sqrt(tf.reduce_sum(tf.square(generated_response),1))
        encoding_utterance_norm = tf.sqrt(tf.reduce_sum(tf.square(encoding_utterance), 1))

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

        #logits=tf.divide(logits,)#
        print('logits shape {}'.format(logits))
        logits = tf.squeeze(logits, [2])  # 删除第三个维度?

        norm_mul = tf.expand_dims(tf.multiply(generated_response_norm, encoding_utterance_norm),1)
        print('norm{}'.format(norm_mul.get_shape()))
        logits = tf.divide(logits,norm_mul )

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
    #return probs, mean_loss
    return logits, mean_loss
