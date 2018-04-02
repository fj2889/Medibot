import tensorflow as tf
import numpy as np
from models import helpers
from models import model

FLAGS = tf.flags.FLAGS


def get_embeddings(hparams):
    if hparams.glove_path and hparams.vocab_path:
        tf.logging.info("Loading Glove embeddings...")
        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        glove_vectors, glove_dict = helpers.load_glove_vectors(
            hparams.glove_path, vocab=set(vocab_array))
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
        RNNInit,
        isBiDirection=False):

    # Initialize embedidngs randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(hparams)
    print('context shape {}'.format(context))  # 顺便显示shape
    # Embed the context and the utterance
    context_embedded = tf.nn.embedding_lookup(  # 三维？
        embeddings_W, context, name="embed_context")
    utterance_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterance, name="embed_utterance")

    print('context_embedded shape {}'.format(context_embedded))
    print('utterence_embedded shape {}'.format(utterance_embedded))

    if model_fun is model.RNN:
        outputs, states = model.RNN(RNNInit,
                                    hparams,
                                    context_embedded,
                                    context_len,
                                    utterance_embedded,
                                    utterance_len,
                                    isBiDirection)
        encoding_context, encoding_utterance = model.process_state(states)
    elif model_fun in set([model.RNN_MaxPooling, model.RNN_CNN_MaxPooling, model.RNN_CNN_MaxPooling, model.RNN_Attention]):
        encoding_context, encoding_utterance = model_fun(RNNInit,
                                                         hparams,
                                                         context_embedded,
                                                         context_len,
                                                         utterance_embedded,
                                                         utterance_len,
                                                         isBiDirection)
    else:
        raise ValueError('model_fun illegal!!!!!!')
    print('encoding_context shape {}'.format(encoding_context))
    print('encoding_utterence shape {}'.format(encoding_utterance))
    with tf.variable_scope("prediction"):
        M = tf.get_variable("M",
                            shape = [encoding_context.get_shape()[1],
                            encoding_context.get_shape()[1]],
                            initializer=tf.truncated_normal_initializer())

        # "Predict" a  response: c * M
        generated_response = tf.matmul(encoding_context, M)
        print('c*M {}'.format(generated_response))
        generated_response = tf.expand_dims(
            generated_response, 2)  # 增加了第三个维度？？
        # 增加维度是为了让每个batch分开做乘积，也就是乘积只做用在后面两个维度
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        print('expand dims generated response {}'.format(generated_response))
        print('expand dims encoding utterence {}'.format(encoding_utterance))
        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response,
                           encoding_utterance, True)  # 维度匹配
        print('logits shape {}'.format(logits))
        logits = tf.squeeze(logits, [2])  # 删除第三个维度?
        print('squeeze logits shape {}'.format(logits))
        # Apply sigmoid to convert logits to probabilities
        probs = tf.sigmoid(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss
# outputs,states=GRU()
    #encoding_context, encoding_utterance=process_state(states,'LSTM')

    #encoding_context, encoding_utterance=RNN_MaxPooling(LSTM)
    #encoding_context, encoding_utterance=RNN_CNN_MaxPooling(Bi_Directional_LSTM,filtersizes=[2,3,4,5],num_filters=30)
    # encoding_context, encoding_utterance=RNN_Attention(Bi_Directional_GRU)'''

    '''outputs,states=RNN(tf.nn.rnn_cell.BasicRNNCell,
                     hparams,
                     context_embedded,
                     context_len,
                     utterance_embedded,
                     utterance_len,
                     True)'''
    #encoding_context, encoding_utterance = process_state(states)
    '''encoding_context, encoding_utterance=RNN_CNN_MaxPooling([2,3,4,5],30,
                                                          tf.nn.rnn_cell.GRUCell,
                                                          hparams,
                                                          context_embedded,
                                                          context_len,
                                                          utterance_embedded,
                                                          utterance_len,
                                                          False)'''


'''encoding_context, encoding_utterance=model.RNN_Attention(tf.nn.rnn_cell.LSTMCell,
                                                          hparams,
                                                          context_embedded,
                                                          context_len,
                                                          utterance_embedded,
                                                          utterance_len,
                                                          True)'''
'''encoding_context,encoding_utterance=RNN_MaxPooling(tf.nn.rnn_cell.LSTMCell,
                     hparams,
                     context_embedded,
                     context_len,
                     utterance_embedded,
                     utterance_len)'''
