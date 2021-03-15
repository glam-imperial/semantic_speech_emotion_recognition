from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def recurrent_model(net, emb=None, hidden_units=256, number_of_outputs=3):
    # net_1 = tf.contrib.layers.fully_connected(net, num_outputs=512)
    # net_2 = tf.contrib.layers.fully_connected(net, num_outputs=512)
    # net_3 = tf.contrib.layers.fully_connected(net, num_outputs=512)

    # Fusing strategies of output of CNN model with word embeddings

    # Attention aprroach
    fused_features = attention_model(net, emb, projected_units=1024)

    net_1 = tf.contrib.layers.fully_connected(fused_features, num_outputs=512)
    net_2 = tf.contrib.layers.fully_connected(fused_features, num_outputs=512)
    net_3 = tf.contrib.layers.fully_connected(fused_features, num_outputs=512)

    # Concat output
    # net = tf.concat([net_1, net_2, net_3], axis=2)

    # Self-attention
    attn_1 = attention_model(net_1, net_2, scope='_self12')
    net = attention_model(attn_1, net_3, scope='_self123')

    #-----------------------------------------#
    # Fully connected layer approach
    # net = fully_connected_model(net, emb)
    print(net)

    with tf.variable_scope('recurrent'):
        batch_size, seq_length, num_features = net.get_shape().as_list()

        # print(batch_size, seq_length, num_features)

        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                       use_peepholes=True,
                                       cell_clip=100,
                                       state_is_tuple=True)

        # stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

        outputs, states = tf.nn.dynamic_rnn(lstm, net, dtype=tf.float32)
        net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
        prediction = tf.nn.tanh(slim.layers.linear(net, number_of_outputs))

        return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))


def audio_model2(audio_frames=None, conv_filters=40):
    with tf.variable_scope('audio_model'):

        batch_size, seq_length, num_features = audio_frames.get_shape().as_list()
        # print(audio_frames.get_shape().as_list())

        audio_input = tf.reshape(audio_frames, [batch_size,  num_features * seq_length, 1])
        # print(audio_input)

        net = tf.layers.conv1d(audio_input, 50, 8, padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling1d(net, 10, 10)
        net = slim.dropout(net, 0.5)
        # print(net)

        net = tf.layers.conv1d(net, 125, 6, padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling1d(net, 5, 5)
        net = slim.dropout(net, 0.5)
        # print(net)

        net = tf.layers.conv1d(net, 250, 6, padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling1d(net, 5, 5)
        net = slim.dropout(net, 0.5)
        # print(net)

        net = tf.reshape(net, [batch_size, seq_length, -1])
        # print(net)

        return net


def fully_connected_model(audio_frames, text_frames):
    with tf.variable_scope('FC_model'):
        audio_features = tf.contrib.layers.fully_connected(audio_frames, num_outputs=1024)
        text_features = tf.contrib.layers.fully_connected(text_frames, num_outputs=1024)

        net = tf.concat([audio_features, text_features], axis=2)
        net = tf.contrib.layers.fully_connected(net, num_outputs=512)
        return net


def attention_model(audio_frames, text_frames, projected_units=2048, scope=''):
    with tf.variable_scope('attn_model' + scope):
        batch_size, seq_length, num_features = audio_frames.get_shape().as_list()

        # Attention
        audio_features = tf.reshape(audio_frames, [-1, audio_frames.get_shape()[-1]])
        text_features = tf.reshape(text_frames, [-1, text_frames.get_shape()[-1]])

        # print(audio_features.get_shape().as_list(), text_features.get_shape().as_list())

        if audio_features.get_shape().as_list()[1] != text_features.get_shape().as_list()[1]:
            projected_audio = tf.contrib.layers.fully_connected(audio_features, num_outputs=projected_units)
            projected_text = tf.contrib.layers.fully_connected(text_features, num_outputs=projected_units)
        else:
            projected_audio = audio_features
            projected_text = text_features

        net = stack_attention_producer(projected_audio, projected_text, batch_size, 'attn')

        return net


def stack_attention_producer(frame1, frame2, batch_size, scope=None):

    with tf.variable_scope(scope):

        frame1 = tf.expand_dims(frame1, 1)
        frame2 = tf.expand_dims(frame2, 1)

        frames = tf.concat([frame1, frame2], axis=1)
        tmp_frames = tf.expand_dims(frames, 3)

        # print(frames, tmp_frames)

        conv = tf.squeeze(conv2d(tmp_frames, 1, 1, frames.get_shape()[-1], name=scope + '_selector'), [2, 3])
        # print(conv)

        conv = tf.multiply(conv, 1. / tf.sqrt(tf.cast(frames.get_shape()[-1], tf.float32)))
        # print(conv)

        attention = tf.nn.softmax(conv, name=scope + '_softmax')
        attention = tf.expand_dims(attention, axis=2)
        # print(attention)

        out = tf.reduce_sum(tf.multiply(frames, attention), 1, keep_dims=True)
        # print(out)

        out = tf.reshape(out, (batch_size, -1, out.shape[-1]))
        return out


def conv2d(input_, output_dim, k_h, k_w, padding='VALID', name='conv2d'):
    """
         :param input_: shape = [batch_size * num_unroll_steps, 1, max_sent_length, embed_size]
         :param output_dim: [kernel_features], which is # of kernels with this width
         :param k_h: 1
         :param k_w: kernel width, n-grams
         :param name: name scope
         :return: shape = [reduced_length, output_dim]
    """

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding=padding) + b


def get_model(name):
    name_to_fun = {'audio_model2': audio_model2}

    if name in name_to_fun:
        model = name_to_fun[name]
    else:
        raise ValueError('Requested name [{}] not a valid model'.format(name))

    def wrapper(*args, **kwargs):
        return recurrent_model(model(*args), **kwargs)

    return wrapper
