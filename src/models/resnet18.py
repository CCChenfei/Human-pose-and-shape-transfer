from collections import namedtuple

import tensorflow as tf
import numpy as np




HParams = namedtuple('HParams',
                    'batch_size, num_gpus, num_classes, weight_decay, '
                     'momentum, finetune')

class ResNet(object):
    def __init__(self, num_classes=1000):


        self._counted_scope = []
        self._flops = 0
        self._weights = 0
        self._num_classes=num_classes


    def build_tower(self, images,reuse):
        # with tf.variable_scope(name,reuse=reuse):
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        feature_maps=[]
        self._reuse_weights=reuse


        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1',reuse=reuse) as scope:
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            variables = tf.contrib.framework.get_variables(scope)

        feature_maps.append(x)


        # conv2_x
        x,var1 = self._residual_block(x, name='conv2_1')
        x,var2 = self._residual_block(x, name='conv2_2')
        variables.extend(var1)
        variables.extend(var2)
        feature_maps.append(x)

        # conv3_x
        x, var1= self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x, var2= self._residual_block(x, name='conv3_2')

        variables.extend(var1)
        variables.extend(var2)
        feature_maps.append(x)

        # conv4_x
        x, var1 = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x, var2 = self._residual_block(x, name='conv4_2')

        variables.extend(var1)
        variables.extend(var2)
        feature_maps.append(x)

        # conv5_x
        x, var1= self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x, var2= self._residual_block(x, name='conv5_2')

        variables.extend(var1)
        variables.extend(var2)
        feature_maps.append(x)

        # Logit
        with tf.variable_scope('logits',reuse=reuse) as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            x = self._fc(x, self._num_classes)
            var1=tf.contrib.framework.get_variables(scope)
        variables.extend(var1)

        logits = x

        # Probs & preds & acc
        probs = tf.nn.softmax(x)
        preds = tf.to_int32(tf.argmax(logits, 1))
        # ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        # zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        # correct = tf.where(tf.equal(preds, labels), ones, zeros)
        # acc = tf.reduce_mean(correct)

        # Loss & acc
        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=labels)
        # loss = tf.reduce_mean(losses)

        return feature_maps,variables








    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name,reuse=self._reuse_weights) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        variables = tf.contrib.framework.get_variables(scope)
        return x,variables


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name,reuse=self._reuse_weights) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        variables = tf.contrib.framework.get_variables(scope)
        return x,variables


    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = _conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = _fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = _bn(x, tf.constant(False,dtype=tf.bool), 0, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = _relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)


import numpy as np
import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _conv(x, filter_size, out_channel, strides, pad='SAME', input_q=None, output_q=None, name='conv'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)

        # Split and split loss
        if (input_q is not None) and (output_q is not None):
            # w = tf.reduce_mean(kernel, axis=[0, 1])
            # w = tf.sqrt(tf.reduce_mean(tf.square(kernel), [0, 1]))
            _add_split_loss(kernel, input_q, output_q)

    return conv


def _fc(x, out_dim, input_q=None, output_q=None, name='fc'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    with tf.variable_scope(name):
        # Main operation: fc
        with tf.device('/CPU:0'):
            w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)

        # Split loss
        if (input_q is not None) and (output_q is not None):
            _add_split_loss(w, input_q, output_q)

    return fc


def _get_split_q(ngroups, dim, name='split', l2_loss=False):
    with tf.variable_scope(name):
        # alpha = tf.get_variable('alpha', shape=[ngroups, dim], dtype=tf.float32,
                              # initializer=tf.random_normal_initializer(stddev=0.1))
        # q = tf.nn.softmax(alpha, dim=0, name='q')
        std_dev = 0.01
        init_val = np.random.normal(0, std_dev, (ngroups, dim))
        init_val = init_val - np.average(init_val, axis=0) + 1.0/ngroups
        with tf.device('/CPU:0'):
            q = tf.get_variable('q', shape=[ngroups, dim], dtype=tf.float32,
                                # initializer=tf.constant_initializer(1.0/ngroups))
                                initializer=tf.constant_initializer(init_val))
        if l2_loss:
            if q not in tf.get_collection(WEIGHT_DECAY_KEY):
                tf.add_to_collection(WEIGHT_DECAY_KEY, q*2.236)

    return q

def _merge_split_q(q, merge_idxs, name='merge'):
    assert len(q.get_shape()) == 2
    ngroups, dim = q.get_shape().as_list()
    assert ngroups == len(merge_idxs)

    with tf.variable_scope(name):
        max_idx = np.max(merge_idxs)
        temp_list = []
        for i in range(max_idx + 1):
            temp = []
            for j in range(ngroups):
                if merge_idxs[j] == i:
                    temp.append(tf.slice(q, [j, 0], [1, dim]))
            temp_list.append(tf.add_n(temp))
        ret = tf.concat(0, temp_list)

    return ret


def _get_even_merge_idxs(N, split):
    assert N >= split
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]


def _add_split_loss(w, input_q, output_q):
    # Check input tensors' measurements
    assert len(w.get_shape()) == 2 or len(w.get_shape()) == 4
    in_dim, out_dim = w.get_shape().as_list()[-2:]
    assert len(input_q.get_shape()) == 2
    assert len(output_q.get_shape()) == 2
    assert in_dim == input_q.get_shape().as_list()[1]
    assert out_dim == output_q.get_shape().as_list()[1]
    assert input_q.get_shape().as_list()[0] == output_q.get_shape().as_list()[0]  # ngroups
    ngroups = input_q.get_shape().as_list()[0]
    assert ngroups > 1

    # Add split losses to collections
    T_list = []
    U_list = []
    if input_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS'):
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', input_q)
        print('\t\tAdd overlap & split loss for %s' % input_q.name)
        for i in range(ngroups):
            for j in range(ngroups):
                if i == j:
                    continue
                T_list.append(tf.reduce_sum(input_q[i,:] * input_q[j,:]))
            U_list.append(tf.square(tf.reduce_sum(input_q[i,:])))
    if output_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS'):
        print('\t\tAdd overlap & split loss for %s' % output_q.name)
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', output_q)
        for i in range(ngroups):
            for j in range(ngroups):
                if i == j:
                    continue
                T_list.append(tf.reduce_sum(output_q[i,:] * output_q[j,:]))
            U_list.append(tf.square(tf.reduce_sum(output_q[i,:])))
    if T_list:
        tf.add_to_collection('OVERLAP_LOSS', tf.add_n(T_list))
    if U_list:
        tf.add_to_collection('UNIFORM_LOSS', tf.add_n(U_list))

    S_list = []
    for i in range(ngroups):
        if len(w.get_shape()) == 4:
            w_reduce = tf.reduce_mean(tf.square(w), [0, 1])
            wg_row = tf.matmul(tf.matmul(tf.diag(tf.square(1 - input_q[i,:])), w_reduce), tf.diag(tf.square(output_q[i,:])))
            wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row, 1)))
            wg_col = tf.matmul(tf.matmul(tf.diag(tf.square(input_q[i,:])), w_reduce), tf.diag(tf.square(1 - output_q[i,:])))
            wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col, 0)))
        else:  # len(w.get_shape()) == 2
            wg_row = tf.matmul(tf.matmul(tf.diag(1 - input_q[i,:]), w), tf.diag(output_q[i,:]))
            wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row * wg_row, 1)))
            wg_col = tf.matmul(tf.matmul(tf.diag(input_q[i,:]), w), tf.diag(1 - output_q[i,:]))
            wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col * wg_col, 0)))
        S_list.append(wg_row_l2 + wg_col_l2)
    S = tf.add_n(S_list)
    tf.add_to_collection('WEIGHT_SPLIT', S)

    # Add histogram for w if split losses are added
    scope_name = tf.get_variable_scope().name
    tf.histogram_summary("%s/weights" % scope_name, w)
    print('\t\tAdd split loss for %s(%dx%d, %d groups)' \
          % (tf.get_variable_scope().name, in_dim, out_dim, ngroups))

    return


def _bn(x, is_train, global_step=None, name='bn'):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
            # decay = moving_average_decay
        # else:
            # decay = tf.cond(tf.greater(global_step, 100)
                            # , lambda: tf.constant(moving_average_decay, tf.float32)
                            # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
                                          # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                          # scale=True, epsilon=1e-5, is_training=is_train,
                                          # trainable=True)
    return bn


## Other helper functions
