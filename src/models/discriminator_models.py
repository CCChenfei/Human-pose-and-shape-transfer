import tensorflow as tf
import tensorflow.contrib.slim as slim


def NLayerDiscriminator(x,name=None,is_training=True,ndf=64,n_layers=3,reuse=False,use_sigmoid=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        x=slim.conv2d(x,ndf,4,2,activation_fn=None)
        x=slim.nn.leaky_relu(x)
        for i in range(1,n_layers):
            nf_mult = min(2 ** i, 8)
            x=slim.conv2d(x,ndf*nf_mult,4,2)
            x=slim.batch_norm(x,is_training=is_training)
            x=tf.nn.leaky_relu(x)
        nf_mult = min(2 ** n_layers, 8)
        x = slim.conv2d(x, ndf * nf_mult, 4, 2)
        x = slim.batch_norm(x,is_training=is_training)
        x = tf.nn.leaky_relu(x)
        x=slim.conv2d(x,1,4,1)
        if use_sigmoid:
            x=slim.nn.sigmoid(x)

    variables = tf.contrib.framework.get_variables(scope)

    return x,variables



#BEGAN
import numpy as np


def DiscriminatorBEGAN(x, name=None,reuse=False,input_channel=3, z_num=64, repeat_num=6, hidden_num=128, data_format='NHWC'):
    def reshape(x, h, w, c, data_format):
        if data_format == 'NCHW':
            x = tf.reshape(x, [-1, c, h, w])
        else:
            x = tf.reshape(x, [-1, h, w, c])
        return x

    def int_shape(tensor):
        shape = tensor.get_shape().as_list()
        return [num if num is not None else -1 for num in shape]

    def resize_nearest_neighbor(x, new_size, data_format):
        if data_format == 'NCHW':
            x = nchw_to_nhwc(x)
            x = tf.image.resize_nearest_neighbor(x, new_size)
            x = nhwc_to_nchw(x)
        else:
            x = tf.image.resize_nearest_neighbor(x, new_size)
        return x

    def get_conv_shape(tensor, data_format):
        shape = int_shape(tensor)
        # always return [N, H, W, C]
        if data_format == 'NCHW':
            return [shape[0], shape[2], shape[3], shape[1]]
        elif data_format == 'NHWC':
            return shape

    def nchw_to_nhwc(x):
        return tf.transpose(x, [0, 2, 3, 1])

    def nhwc_to_nchw(x):
        return tf.transpose(x, [0, 3, 1, 2])

    def upscale(x, scale, data_format):
        _, h, w, _ = get_conv_shape(x, data_format)
        return resize_nearest_neighbor(x, (h * scale, w * scale), data_format)

    with tf.variable_scope(name,reuse=reuse) as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)

        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables


def StarDiscriminator(x,name=None,ndf=64,n_layers=6,c_dim=3,reuse=False):
    def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
        weight_init = tf.contrib.layers.xavier_initializer()
        weight_regularizer = None
        with tf.variable_scope(scope):
            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

            return x

    def lrelu(x, alpha=0.2):
        return tf.nn.leaky_relu(x, alpha)
    with tf.variable_scope(name, reuse=reuse) as scope:
        channel = ndf
        img_size=x.shape[1].value
        x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_0')
        x = lrelu(x, 0.01)

        for i in range(1, n_layers):
            x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_' + str(i))
            x = lrelu(x, 0.01)

            channel = channel * 2

        c_kernel = int(img_size / np.power(2, n_layers))

        logit = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')
        c = conv(x, channels=c_dim, kernel=c_kernel, stride=1, use_bias=False, scope='D_label')
        c = tf.reshape(c, shape=[-1, c_dim])

    variables = tf.contrib.framework.get_variables(scope)

    return logit,variables