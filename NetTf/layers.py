import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops

# 自归一化激活函数，效果相当于 ReLU+BN
def selu(x, name='selu'):
    with ops.name_scope(name) as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))

def c1rb(net, filters, kernel_size,training,dilation_rate=1, activation=True, scope=None):

    with tf.variable_scope(scope):

        kernal_units = kernel_size * net.shape.as_list()[-1]

        net = tf.layers.conv1d(net, filters, kernel_size,
                               padding='same',
                               dilation_rate=dilation_rate,
                               activation=None,
                               use_bias=True,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=np.sqrt(1/kernal_units)),
                               trainable=training,
                               name='gconv')

        if activation:
            net = selu(net, name='selu')

        return net

def c2rb(net, filters, kernel_size,training=True,padding='same',dilation_rate=1, activation=True, scope=None):

    with tf.variable_scope(scope):

        kernal_units = kernel_size[0] * kernel_size[1] * net.shape.as_list()[-1]

        net = tf.layers.conv2d(net, filters, kernel_size,
                               padding=padding,
                               dilation_rate=dilation_rate,
                               activation=None,
                               use_bias=True,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=np.sqrt(1/kernal_units)),
                               trainable=training,
                               name='conv')

        if activation:
            net = selu(net, name='selu')

        return net

def _upscore_layer_1d(input, shape, stride, name, num_class,reuse,filter):
    with tf.variable_scope(name, reuse) as scope:
        # 最后的上采样最后一维为类别数 X 上采样的最后一维都为类别数，做融合之前的那次也是一样
        input=tf.expand_dims(input,axis=2)
        out_channels = num_class

        output_shape = [shape[0], shape[1], 1, out_channels]
        output_shape = tf.stack(output_shape)

        filter = tf.cast(filter, tf.float32)

        upscore = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape,
                                         strides=[1, stride, 1, 1],
                                         padding='SAME', name='upscore')
        return tf.squeeze(upscore,axis=2)

def _upscore_layer(input, shape, ksize, stride, name, num_class,reuse=False, stddev=5e-4, wd=0):
    with tf.variable_scope(name, reuse) as scope:
        # 最后的上采样最后一维为类别数 X 上采样的最后一维都为类别数，做融合之前的那次也是一样
        in_channels = input.get_shape()[3].value
        out_channels = num_class

        f_shape = [ksize, ksize, out_channels, in_channels]

        filter = get_conv_filter(f_shape,stddev, wd)
        #filter = get_deconv_filter(f_shape, wd)

        output_shape = [shape[0], shape[1], shape[2], out_channels]
        output_shape = tf.stack(output_shape)

        filter = tf.cast(filter, tf.float32)

        upscore = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME', name='upscore')
        #tf.summary.histogram('upscore_layer', upscore)
        return upscore

def _upscore_layer2(input, num_class, ksize, stride, scope,training):
    with tf.variable_scope(scope):
        upscore = tf.layers.conv2d_transpose(input, filters=num_class, kernel_size=ksize,
                                             strides=stride, padding='same',
                                             kernel_initializer=tf.ones_initializer(),
                                             trainable=training,
                                             name='upscore')

        return upscore

# 获取卷积参数
def get_conv_filter(shape, stddev, wd):

    weights = tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=stddev, name='conv_weights'))
    #adding weight decay
    if not tf.get_variable_scope().reuse:
         weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
         tf.add_to_collection('losses', weight_decay)

    return weights

def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret

try:
    @ops.RegisterGradient("MaxPoolWithArgmax")
    def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
        return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                     grad,
                                                     op.outputs[1],
                                                     op.get_attr("ksize"),
                                                     op.get_attr("strides"),
                                                     padding=op.get_attr("padding"))
except Exception as e:
    print(f"Could not add gradient for MaxPoolWithArgMax, Likely installed already (tf 1.4)")
    print(e)