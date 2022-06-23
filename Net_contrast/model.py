import tensorflow as tf
from NetTf.layers import unpool_with_argmax,c2rb,_upscore_layer

def inference_segnet(images, class_inc_bg=None):
    with tf.variable_scope('pool1'):
        net1 = c2rb(images, 64, [3, 3], scope='conv1_1')
        net1 = c2rb(net1, 64, [3, 3],  scope='conv1_2')
        net1, arg1 = tf.nn.max_pool_with_argmax(net1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')

    with tf.variable_scope('pool2'):
        net2 = c2rb(net1, 128, [3, 3], scope='conv2_1')
        net2 = c2rb(net2, 128, [3, 3], scope='conv2_2')
        net2 = c2rb(net2, 128, [3, 3], scope='conv2_3')
        net2, arg2 = tf.nn.max_pool_with_argmax(net2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')

    with tf.variable_scope('pool3'):
        net3 = c2rb(net2, 256, [3, 3], scope='conv3_1')
        net3 = c2rb(net3, 256, [3, 3], scope='conv3_2')
        net3 = c2rb(net3, 256, [3, 3], scope='conv3_3')
        net3, arg3 = tf.nn.max_pool_with_argmax(net3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool3')

    with tf.variable_scope('pool4'):
        net4 = c2rb(net3, 512, [3, 3], scope='conv4_1')
        net4 = c2rb(net4, 512, [3, 3], scope='conv4_2')
        net4 = c2rb(net4, 512, [3, 3], scope='conv4_3')
        net4, arg4 = tf.nn.max_pool_with_argmax(net4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    with tf.variable_scope('pool5'):
        net5 = c2rb(net4, 512, [3, 3], scope='conv5_1')
        net5 = c2rb(net5, 512, [3, 3], scope='conv5_2')
        net5 = c2rb(net5, 512, [3, 3], scope='conv5_3')
        net5, arg5 = tf.nn.max_pool_with_argmax(net5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool5')

    with tf.variable_scope('unpool5'):
        net6 = unpool_with_argmax(net5, arg5, name='maxunpool5')
        net6 = c2rb(net6, 512, [3, 3], scope='uconv5_3')
        net6 = c2rb(net6, 512, [3, 3], scope='uconv5_2')
        net6 = c2rb(net6, 512, [3, 3], scope='uconv5_1')

        #combine low level and high level information
        net4=c2rb(net4,512,[1,1],scope='conv6')
        net6=tf.add(net4,net6)

    with tf.variable_scope('unpool4'):
        net7 = unpool_with_argmax(net6, arg4, name='maxunpool4')
        net7 = c2rb(net7, 512, [3, 3], scope='uconv4_3')
        net7 = c2rb(net7, 512, [3, 3], scope='uconv4_2')
        net7 = c2rb(net7, 256, [3, 3], scope='uconv4_1')

        # combine low level and high level information
        net3 = c2rb(net3, 256, [1, 1], scope='conv7')
        net7 = tf.add(net3, net7)

    with tf.variable_scope('unpool3'):
        net8 = unpool_with_argmax(net7, arg3, name='maxunpool3')
        net8 = c2rb(net8, 256, [3, 3], scope='uconv3_3')
        net8 = c2rb(net8, 256, [3, 3], scope='uconv3_2')
        net8 = c2rb(net8, 128, [3, 3], scope='uconv3_1')

        # combine low level and high level information
        net2 = c2rb(net2, 128, [1, 1], scope='conv8')
        net8 = tf.add(net2, net8)

    with tf.variable_scope('unpool2'):
        net9 = unpool_with_argmax(net8, arg2, name='maxunpool2')
        net9 = c2rb(net9, 128, [3, 3], scope='uconv2_3')
        net9 = c2rb(net9, 128, [3, 3], scope='uconv2_2')
        net9 = c2rb(net9, 64, [3, 3], scope='uconv2_1')

    with tf.variable_scope('unpool1'):
        net10 = unpool_with_argmax(net9, arg1, name='maxunpool1')
        net10 = c2rb(net10, 64, [3, 3], scope='uconv1_2')

    with tf.variable_scope('output'):
        logits = c2rb(net10, class_inc_bg, [3, 3], activation=False, scope='logits')
        softmax_logits = tf.nn.softmax(logits=logits, dim=3, name='softmax_logits')

    return logits, softmax_logits

#2D Unet 网络
def inference_2D_unet(images, num_classes):
    with tf.variable_scope('pool1'):
        conv1_1 = c2rb(images, 64, [3, 3], scope='conv1_1')
        conv1_2 = c2rb(conv1_1, 64, [3, 3], scope='conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool2'):
        conv2_1 = c2rb(pool1, 128, [3, 3], scope='conv2_1')
        conv2_2 = c2rb(conv2_1, 128, [3, 3], scope='conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool3'):
        conv3_1 = c2rb(pool2, 256, [3, 3], scope='conv3_1')
        conv3_2 = c2rb(conv3_1, 256, [3, 3], scope='conv3_2')
        pool3 = tf.layers.max_pooling2d(conv3_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool4'):
        conv4_1 = c2rb(pool3, 512, [3, 3], scope='conv4_1')
        conv4_2 = c2rb(conv4_1, 512, [3, 3], scope='conv4_2')
        pool4 = tf.layers.max_pooling2d(conv4_2, pool_size=(2, 2), strides=(2, 2), padding='same')

    with tf.variable_scope('vally'):
        vally1 = c2rb(pool4, 1024, [3, 3], scope='vally1')
        vally2 = c2rb(vally1, 1024, [3, 3], scope='vally2')

    with tf.variable_scope('up6'):
        up6_1 = _upscore_layer(vally2, shape=conv4_2.shape.as_list(), ksize=4, stride=2, name='up6',
                               num_class=vally2.shape.as_list()[-1])
        up6_2 = c2rb(up6_1, 512, [3, 3], scope='up6_2')
        concat6 = tf.concat([up6_2, conv4_2], axis=3)
        up6_3 = c2rb(concat6, 512, [3, 3], scope='up6_3')
        up6_4 = c2rb(up6_3, 512, [3, 3], scope='up6_4')

    with tf.variable_scope('up7'):
        up7_1 = _upscore_layer(up6_4, shape=conv3_2.shape.as_list(), ksize=4, stride=2, name='up7',
                               num_class=up6_4.shape.as_list()[-1])
        up7_2 = c2rb(up7_1, 256, [3, 3], scope='up7_2')
        concat7 = tf.concat([up7_2, conv3_2], axis=3)
        up7_3 = c2rb(concat7, 256, [3, 3], scope='up7_3')
        up7_4 = c2rb(up7_3, 256, [3, 3], scope='up7_4')

    with tf.variable_scope('up8'):
        up8_1 = _upscore_layer(up7_4, shape=conv2_2.shape.as_list(), ksize=4, stride=2, name='up8',
                               num_class=up7_4.shape.as_list()[-1])
        up8_2 = c2rb(up8_1, 128, [3, 3], scope='up8_2')
        concat8 = tf.concat([up8_2, conv2_2], axis=3)
        up8_3 = c2rb(concat8, 128, [3, 3], scope='up8_3')
        up8_4 = c2rb(up8_3, 128, [3, 3], scope='up8_4')

    with tf.variable_scope('up9'):
        up9_1 = _upscore_layer(up8_4, shape=conv1_2.shape.as_list(), ksize=4, stride=2, name='up9',
                               num_class=up8_4.shape.as_list()[-1])
        up9_2 = c2rb(up9_1, 64, [3, 3], scope='up8_2')
        concat9 = tf.concat([up9_2, conv1_2], axis=3)
        up9_3 = c2rb(concat9, 64, [3, 3], scope='up8_3')
        up9_4 = c2rb(up9_3, 64, [3, 3], scope='up8_4')

    conv10 = c2rb(up9_4, num_classes, [3, 3], scope='conv10')
    softmax_conv10 = tf.nn.softmax(conv10, dim=3, name ='softmax_conv10')

    return conv10, softmax_conv10

#DDCNN
def inference_ddcnn(images, num_classes):
    with tf.variable_scope('dilation_conv'):
        atrous_conv1_1 = c2rb(images, 64, [3, 3], dilation_rate=1, scope='d_conv1')
        atrous_conv2_1 = c2rb(images, 128, [3, 3], dilation_rate=2, scope='d_conv2')
        atrous_conv3_1 = c2rb(images, 128, [3, 3], dilation_rate=4, scope='d_conv3')
        atrous_conv4_1 = c2rb(images, 128, [3, 3], dilation_rate=8, scope='d_conv4')

        atrous_conv1_2 = tf.layers.max_pooling2d(atrous_conv1_1, pool_size=(3, 3), strides=(1, 1), padding='same', name='pooling')
        atrous_conv2_2 = tf.layers.max_pooling2d(atrous_conv2_1, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')
        atrous_conv3_2 = tf.layers.max_pooling2d(atrous_conv3_1, pool_size=(3, 3), strides=(4, 4), padding='same', name='pooling')
        atrous_conv4_2 = tf.layers.max_pooling2d(atrous_conv4_1, pool_size=(3, 3), strides=(8, 8), padding='same', name='pooling')

    with tf.variable_scope('pool1'):
        conv1_1 = c2rb(atrous_conv1_2, 64, [3, 3], scope='conv1_1')
        conv1_2 = c2rb(conv1_1, 64, [3, 3], scope='conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool2'):
        conv2_0=tf.concat([atrous_conv2_2,pool1],axis=-1)
        conv2_1 = c2rb(conv2_0, 128, [3, 3], scope='conv2_1')
        conv2_2 = c2rb(conv2_1, 128, [3, 3], scope='conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool3'):
        conv3_0 = tf.concat([atrous_conv3_2, pool2], axis=-1)
        conv3_1 = c2rb(conv3_0, 256, [3, 3], scope='conv3_1')
        conv3_2 = c2rb(conv3_1, 256, [3, 3], scope='conv3_2')
        conv3_3 = c2rb(conv3_2, 256, [3, 3], scope='conv3_3')
        pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool4'):
        conv4_0 = tf.concat([atrous_conv4_2, pool3], axis=-1)
        conv4_1 = c2rb(conv4_0, 512, [3, 3], scope='conv4_1')
        conv4_2 = c2rb(conv4_1, 512, [3, 3], scope='conv4_2')
        conv4_3 = c2rb(conv4_2, 512, [3, 3], scope='conv4_3')
        pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=(3, 3), strides=(2, 2), padding='same')

    with tf.variable_scope('pool5'):
        conv5_1 = c2rb(pool4, 512, [3, 3], scope='conv4_1')
        conv5_2 = c2rb(conv5_1, 512, [3, 3], scope='conv4_2')
        conv5_3 = c2rb(conv5_2, 512, [3, 3], scope='conv4_3')
        pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=(3, 3), strides=(1, 1), padding='same')

    with tf.variable_scope('fc6'):
        fc6_1 = c2rb(pool5, 1024, [1, 1], dilation_rate=6, scope='fc6_1')
        fc6_2 = c2rb(pool5, 1024, [1, 1], dilation_rate=12, scope='fc6_2')
        fc6_3 = c2rb(pool5, 1024, [1, 1], dilation_rate=18, scope='fc6_3')
        fc6_4 = c2rb(pool5, 1024, [1, 1], dilation_rate=24, scope='fc6_4')

    with tf.variable_scope('fc7'):
        fc7_1 = c2rb(fc6_1, 1024, [1, 1], scope='fc7_1')
        fc7_2 = c2rb(fc6_2, 1024, [1, 1], scope='fc7_2')
        fc7_3 = c2rb(fc6_3, 1024, [1, 1], scope='fc7_3')
        fc7_4 = c2rb(fc6_4, 1024, [1, 1], scope='fc7_4')

    with tf.variable_scope('fc8'):
        fc8_1 = c2rb(fc7_1, num_classes, [1, 1], scope='fc8_1')
        fc8_2 = c2rb(fc7_2, num_classes, [1, 1], scope='fc8_2')
        fc8_3 = c2rb(fc7_3, num_classes, [1, 1], scope='fc8_3')
        fc8_4 = c2rb(fc7_4, num_classes, [1, 1], scope='fc8_4')

    with tf.variable_scope('up'):
        #求和，双线性插值
        fc=fc8_1+fc8_2+fc8_3+fc8_4

    conv10 = tf.image.resize_bilinear(fc,[320,352],name="bilinear")
    softmax_conv10 = tf.nn.softmax(conv10, dim=3, name ='softmax_conv10')

    return conv10, softmax_conv10

#DD resnet
def inference_dd_resnet(images, num_classes):
    with tf.variable_scope('dilation_conv'):
        atrous_conv1_1 = c2rb(images, 64, [3, 3], dilation_rate=4, scope='d_conv1')
        atrous_conv2_1 = c2rb(images, 64, [3, 3], dilation_rate=1, scope='d_conv2')
        atrous_conv3_1 = c2rb(images, 64, [3, 3], dilation_rate=2, scope='d_conv3')
        atrous_conv4_1 = c2rb(images, 64, [3, 3], dilation_rate=8, scope='d_conv4')

        atrous_conv1_2 = tf.layers.max_pooling2d(atrous_conv1_1, pool_size=(3, 3), strides=(2, 2), padding='same',name='pooling')
        atrous_conv2_2 = tf.layers.max_pooling2d(atrous_conv2_1, pool_size=(3, 3), strides=(2, 2), padding='same',name='pooling')
        atrous_conv3_2 = tf.layers.max_pooling2d(atrous_conv3_1, pool_size=(3, 3), strides=(2, 2), padding='same',name='pooling')
        atrous_conv4_2 = tf.layers.max_pooling2d(atrous_conv4_1, pool_size=(3, 3), strides=(2, 2), padding='same',name='pooling')

    with tf.variable_scope('pool1'):
        conv1_0=tf.concat([atrous_conv1_2,atrous_conv2_2,atrous_conv3_2,atrous_conv4_2],axis=-1)
        conv1_1 = c2rb(conv1_0, 64, [7, 7], scope='conv1_1')
        pool1 = tf.layers.max_pooling2d(conv1_1, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool2'):
        conv2_1_0=c2rb(pool1, 256, [1, 1], scope='conv2_1_0')
        conv2_1_1 = c2rb(conv2_1_0, 64, [1, 1], scope='conv2_1_1')
        conv2_1_2 = c2rb(conv2_1_1, 64, [3, 3], scope='conv2_1_2')
        conv2_1_3 = c2rb(conv2_1_2, 256, [1, 1], scope='conv2_1_3')
        conv2_1_4=conv2_1_0+conv2_1_3

        conv2_2_1 = c2rb(conv2_1_4, 64, [1, 1], scope='conv2_2_1')
        conv2_2_2 = c2rb(conv2_2_1, 64, [3, 3], scope='conv2_2_2')
        conv2_2_3 = c2rb(conv2_2_2, 256, [1, 1], scope='conv2_2_3')
        conv2_2_4 = conv2_1_4 + conv2_2_3

        conv2_3_1 = c2rb(conv2_2_4, 64, [1, 1], scope='conv2_3_1')
        conv2_3_2 = c2rb(conv2_3_1, 64, [3, 3], scope='conv2_3_2')
        conv2_3_3 = c2rb(conv2_3_2, 256, [1, 1], scope='conv2_3_3')
        conv2_3_4 = conv2_2_4 + conv2_3_3
        pool2 = tf.layers.max_pooling2d(conv2_3_4, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool3'):
        conv3_1_0 = c2rb(pool2, 512, [1, 1], scope='conv3_1_0')
        conv3_1_1 = c2rb(conv3_1_0, 128, [1, 1], scope='conv3_1_1')
        conv3_1_2 = c2rb(conv3_1_1, 128, [3, 3], scope='conv3_1_2')
        conv3_1_3 = c2rb(conv3_1_2, 512, [1, 1], scope='conv3_1_3')
        conv3_1_4 = conv3_1_0 + conv3_1_3

        conv3_2_1 = c2rb(conv3_1_4, 128, [1, 1], scope='conv3_2_1')
        conv3_2_2 = c2rb(conv3_2_1, 128, [3, 3], scope='conv3_2_2')
        conv3_2_3 = c2rb(conv3_2_2, 512, [1, 1], scope='conv3_2_3')
        conv3_2_4 = conv3_1_4 + conv3_2_3

        conv3_3_1 = c2rb(conv3_2_4, 128, [1, 1], scope='conv3_3_1')
        conv3_3_2 = c2rb(conv3_3_1, 128, [3, 3], scope='conv3_3_2')
        conv3_3_3 = c2rb(conv3_3_2, 512, [1, 1], scope='conv3_3_3')
        conv3_3_4 = conv3_2_4 + conv3_3_3

        conv3_4_1 = c2rb(conv3_3_4, 128, [1, 1], scope='conv3_4_1')
        conv3_4_2 = c2rb(conv3_4_1, 128, [3, 3], scope='conv3_4_2')
        conv3_4_3 = c2rb(conv3_4_2, 512, [1, 1], scope='conv3_4_3')
        conv3_4_4 = conv3_3_4 + conv3_4_3

        pool3 = tf.layers.max_pooling2d(conv3_4_4, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool4'):
        conv4_1_0 = c2rb(pool3, 1024, [1, 1], scope='conv4_1_0')
        conv4_1_1 = c2rb(conv4_1_0, 256, [1, 1], scope='conv4_1_1')
        conv4_1_2 = c2rb(conv4_1_1, 256, [3, 3], scope='conv4_1_2')
        conv4_1_3 = c2rb(conv4_1_2, 1024, [1, 1], scope='conv4_1_3')
        conv4_1_4 = conv4_1_0 + conv4_1_3

        conv4_2_1 = c2rb(conv4_1_4, 256, [1, 1], scope='conv4_2_1')
        conv4_2_2 = c2rb(conv4_2_1, 256, [3, 3], scope='conv4_2_2')
        conv4_2_3 = c2rb(conv4_2_2, 1024, [1, 1], scope='conv4_2_3')
        conv4_2_4 = conv4_1_4 + conv4_2_3

        conv4_3_1 = c2rb(conv4_2_4, 256, [1, 1], scope='conv4_3_1')
        conv4_3_2 = c2rb(conv4_3_1, 256, [3, 3], scope='conv4_3_2')
        conv4_3_3 = c2rb(conv4_3_2, 1024, [1, 1], scope='conv4_3_3')
        conv4_3_4 = conv4_2_4 + conv4_3_3

        conv4_4_1 = c2rb(conv4_3_4, 256, [1, 1], scope='conv4_4_1')
        conv4_4_2 = c2rb(conv4_4_1, 256, [3, 3], scope='conv4_4_2')
        conv4_4_3 = c2rb(conv4_4_2, 1024, [1, 1], scope='conv4_4_3')
        conv4_4_4 = conv4_2_4 + conv4_4_3

        conv4_5_1 = c2rb(conv4_4_4, 256, [1, 1], scope='conv4_5_1')
        conv4_5_2 = c2rb(conv4_5_1, 256, [3, 3], scope='conv4_5_2')
        conv4_5_3 = c2rb(conv4_5_2, 1024, [1, 1], scope='conv4_5_3')
        conv4_5_4 = conv4_4_4 + conv4_5_3

        conv4_6_1 = c2rb(conv4_5_4, 256, [1, 1], scope='conv4_6_1')
        conv4_6_2 = c2rb(conv4_6_1, 256, [3, 3], scope='conv4_6_2')
        conv4_6_3 = c2rb(conv4_6_2, 1024, [1, 1], scope='conv4_6_3')
        conv4_6_4 = conv4_5_4 + conv4_6_3

        conv4_7_1 = c2rb(conv4_6_4, 256, [1, 1], scope='conv4_7_1')
        conv4_7_2 = c2rb(conv4_7_1, 256, [3, 3], scope='conv4_7_2')
        conv4_7_3 = c2rb(conv4_7_2, 1024, [1, 1], scope='conv4_7_3')
        conv4_7_4 = conv4_6_4 + conv4_7_3

        conv4_8_1 = c2rb(conv4_7_4, 256, [1, 1], scope='conv4_8_1')
        conv4_8_2 = c2rb(conv4_8_1, 256, [3, 3], scope='conv4_8_2')
        conv4_8_3 = c2rb(conv4_8_2, 1024, [1, 1], scope='conv4_8_3')
        conv4_8_4 = conv4_7_4 + conv4_8_3

        conv4_9_1 = c2rb(conv4_8_4, 256, [1, 1], scope='conv4_9_1')
        conv4_9_2 = c2rb(conv4_9_1, 256, [3, 3], scope='conv4_9_2')
        conv4_9_3 = c2rb(conv4_9_2, 1024, [1, 1], scope='conv4_9_3')
        conv4_9_4 = conv4_8_4 + conv4_9_3

        conv4_10_1 = c2rb(conv4_9_4, 256, [1, 1], scope='conv4_10_1')
        conv4_10_2 = c2rb(conv4_10_1, 256, [3, 3], scope='conv4_10_2')
        conv4_10_3 = c2rb(conv4_10_2, 1024, [1, 1], scope='conv4_10_3')
        conv4_10_4 = conv4_9_4 + conv4_10_3

        conv4_11_1 = c2rb(conv4_10_4, 256, [1, 1], scope='conv4_11_1')
        conv4_11_2 = c2rb(conv4_11_1, 256, [3, 3], scope='conv4_11_2')
        conv4_11_3 = c2rb(conv4_11_2, 1024, [1, 1], scope='conv4_11_3')
        conv4_11_4 = conv4_10_4 + conv4_11_3

        conv4_12_1 = c2rb(conv4_11_4, 256, [1, 1], scope='conv4_12_1')
        conv4_12_2 = c2rb(conv4_12_1, 256, [3, 3], scope='conv4_12_2')
        conv4_12_3 = c2rb(conv4_12_2, 1024, [1, 1], scope='conv4_12_3')
        conv4_12_4 = conv4_11_4 + conv4_12_3

        conv4_13_1 = c2rb(conv4_12_4, 256, [1, 1], scope='conv4_13_1')
        conv4_13_2 = c2rb(conv4_13_1, 256, [3, 3], scope='conv4_13_2')
        conv4_13_3 = c2rb(conv4_13_2, 1024, [1, 1], scope='conv4_13_3')
        conv4_13_4 = conv4_12_4 + conv4_13_3

        conv4_14_1 = c2rb(conv4_13_4, 256, [1, 1], scope='conv4_14_1')
        conv4_14_2 = c2rb(conv4_14_1, 256, [3, 3], scope='conv4_14_2')
        conv4_14_3 = c2rb(conv4_14_2, 1024, [1, 1], scope='conv4_14_3')
        conv4_14_4 = conv4_13_4 + conv4_14_3

        conv4_15_1 = c2rb(conv4_14_4, 256, [1, 1], scope='conv4_15_1')
        conv4_15_2 = c2rb(conv4_15_1, 256, [3, 3], scope='conv4_15_2')
        conv4_15_3 = c2rb(conv4_15_2, 1024, [1, 1], scope='conv4_15_3')
        conv4_15_4 = conv4_14_4 + conv4_15_3

        conv4_16_1 = c2rb(conv4_15_4, 256, [1, 1], scope='conv4_16_1')
        conv4_16_2 = c2rb(conv4_16_1, 256, [3, 3], scope='conv4_16_2')
        conv4_16_3 = c2rb(conv4_16_2, 1024, [1, 1], scope='conv4_16_3')
        conv4_16_4 = conv4_15_4 + conv4_16_3

        conv4_17_1 = c2rb(conv4_16_4, 256, [1, 1], scope='conv4_17_1')
        conv4_17_2 = c2rb(conv4_17_1, 256, [3, 3], scope='conv4_17_2')
        conv4_17_3 = c2rb(conv4_17_2, 1024, [1, 1], scope='conv4_17_3')
        conv4_17_4 = conv4_16_4 + conv4_17_3

        conv4_18_1 = c2rb(conv4_17_4, 256, [1, 1], scope='conv4_18_1')
        conv4_18_2 = c2rb(conv4_18_1, 256, [3, 3], scope='conv4_18_2')
        conv4_18_3 = c2rb(conv4_18_2, 1024, [1, 1], scope='conv4_18_3')
        conv4_18_4 = conv4_17_4 + conv4_18_3

        conv4_19_1 = c2rb(conv4_18_4, 256, [1, 1], scope='conv4_19_1')
        conv4_19_2 = c2rb(conv4_19_1, 256, [3, 3], scope='conv4_19_2')
        conv4_19_3 = c2rb(conv4_19_2, 1024, [1, 1], scope='conv4_19_3')
        conv4_19_4 = conv4_18_4 + conv4_19_3

        conv4_20_1 = c2rb(conv4_19_4, 256, [1, 1], scope='conv4_20_1')
        conv4_20_2 = c2rb(conv4_20_1, 256, [3, 3], scope='conv4_20_2')
        conv4_20_3 = c2rb(conv4_20_2, 1024, [1, 1], scope='conv4_20_3')
        conv4_20_4 = conv4_19_4 + conv4_20_3

        conv4_21_1 = c2rb(conv4_20_4, 256, [1, 1], scope='conv4_21_1')
        conv4_21_2 = c2rb(conv4_21_1, 256, [3, 3], scope='conv4_21_2')
        conv4_21_3 = c2rb(conv4_21_2, 1024, [1, 1], scope='conv4_21_3')
        conv4_21_4 = conv4_20_4 + conv4_21_3

        conv4_22_1 = c2rb(conv4_21_4, 256, [1, 1], scope='conv4_22_1')
        conv4_22_2 = c2rb(conv4_22_1, 256, [3, 3], scope='conv4_22_2')
        conv4_22_3 = c2rb(conv4_22_2, 1024, [1, 1], scope='conv4_22_3')
        conv4_22_4 = conv4_21_4 + conv4_22_3

        conv4_23_1 = c2rb(conv4_22_4, 256, [1, 1], scope='conv4_23_1')
        conv4_23_2 = c2rb(conv4_23_1, 256, [3, 3], scope='conv4_23_2')
        conv4_23_3 = c2rb(conv4_23_2, 1024, [1, 1], scope='conv4_23_3')
        conv4_23_4 = conv4_22_4 + conv4_23_3

        #pool4 = tf.layers.max_pooling2d(conv4_23_4, pool_size=(3, 3), strides=(2, 2), padding='same', name='pooling')

    with tf.variable_scope('pool5'):
        conv5_1_0 = c2rb(conv4_23_4, 1024, [1, 1], scope='conv4_1_0')
        conv5_1_1 = c2rb(conv5_1_0, 512, [1, 1], scope='conv5_1_1')
        conv5_1_2 = c2rb(conv5_1_1, 512, [3, 3], scope='conv5_1_2')
        conv5_1_3 = c2rb(conv5_1_2, 1024, [1, 1], scope='conv5_1_3')
        conv5_1_4 = conv5_1_0 + conv5_1_3

        conv5_2_1 = c2rb(conv5_1_4, 512, [1, 1], scope='conv5_2_1')
        conv5_2_2 = c2rb(conv5_2_1, 512, [3, 3], scope='conv5_2_2')
        conv5_2_3 = c2rb(conv5_2_2, 1024, [1, 1], scope='conv5_2_3')
        conv5_2_4 = conv5_1_4 + conv5_2_3

        conv5_3_1 = c2rb(conv5_2_4, 512, [1, 1], scope='conv5_3_1')
        conv5_3_2 = c2rb(conv5_3_1, 512, [3, 3], scope='conv5_3_2')
        conv5_3_3 = c2rb(conv5_3_2, 1024, [1, 1], scope='conv5_3_3')
        conv5_3_4 = conv5_2_4 + conv5_3_3

        pool5 = tf.layers.average_pooling2d(conv5_3_4, pool_size=(3, 3), strides=(1, 1), padding='same', name='pooling')

    with tf.variable_scope('fc6'):
        fc6_1 = c2rb(pool5, num_classes, [1, 1], dilation_rate=6, scope='fc6_1')
        fc6_2 = c2rb(pool5, num_classes, [1, 1], dilation_rate=12, scope='fc6_2')
        fc6_3 = c2rb(pool5, num_classes, [1, 1], dilation_rate=18, scope='fc6_3')
        fc6_4 = c2rb(pool5, num_classes, [1, 1], dilation_rate=24, scope='fc6_4')

    with tf.variable_scope('up'):
        # 求和，双线性插值
        fc = fc6_1 + fc6_2 + fc6_3 + fc6_4

    conv10 = tf.image.resize_bilinear(fc, [320, 352], name="bilinear")
    softmax_conv10 = tf.nn.softmax(conv10, dim=3, name='softmax_conv10')

    return conv10, softmax_conv10