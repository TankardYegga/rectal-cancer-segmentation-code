import tensorflow as tf
from NetTf.layers import c1rb,c2rb,_upscore_layer_1d,_upscore_layer,get_conv_filter

def BPB(img,is_training,i,reuse):
    with tf.variable_scope('g_bpb_'+i, reuse=reuse):
        atrous_conv1 = c2rb(img, 64, [1, 1], is_training, scope='g_bpb1' + i)
        atrous_conv2 = c2rb(img, 64, [3, 3], is_training, scope='g_bpb2' + i)
        atrous_conv3 = c2rb(img, 64, [3, 3], is_training, dilation_rate=2, scope='g_bpb3' + i)
        atrous_conv4 = c2rb(img, 64, [3, 3], is_training, dilation_rate=4, scope='g_bpb4' + i)
        atrous_conv5 = c2rb(img, 64, [3, 3], is_training, dilation_rate=6, scope='g_bpb5' + i)
        atrous_conv = tf.concat([atrous_conv1, atrous_conv2, atrous_conv3, atrous_conv4, atrous_conv5], axis=-1)
        atrous_conv = c2rb(atrous_conv, 1, [1, 1], is_training,activation=False, scope='g_bpb6' + i)

        result = tf.add(img, tf.multiply(img, tf.sigmoid(atrous_conv)))
    return atrous_conv,result

#根据boundary产生对应节点初步坐标,node geneartor，通过pooling的方式
def NG_multilayer_index(img,features,params,is_training,i,reuse):
    with tf.variable_scope('g_ng_'+i, reuse=reuse):
        #boundary map直接转成坐标
        img=img[:,:,:,-1]
        batch = img.shape.as_list()[0]
        width = img.shape.as_list()[2]

        img=tf.reshape(img,[batch,-1])

        #node = tf.nn.top_k(img, k=params['N'], sorted=True) #不是对是否是边界节点置信度值进行排序，而是坐标排序
        node = tf.nn.top_k(img, k=params['N']*2)

        # 反卷积变量的初始化
        filter1 = get_conv_filter([4, 1, 1024, 1024], stddev=5e-4, wd=0)
        filter2 = get_conv_filter([4, 1, 512, 512], stddev=5e-4, wd=0)
        filter3 = get_conv_filter([4, 1, 256, 256], stddev=5e-4, wd=0)

        #当前slice不存在边界
        def true_proc():
            node = tf.constant(params['H']//2,shape=[batch, params['N'], 2],dtype=tf.float32)
            return node

        #当前slice存在边界
        def false_proc():
            img1=1-tf.abs(img-0.5)
            ind=tf.nn.top_k(img1, k=params['N']*2)
            ind = tf.cast(tf.expand_dims(ind.indices, -1), tf.float32)
            ind = tf.concat([ind // width, ind % width], axis=-1)

            # 对节点进行排序
            #ind = ind[:, :, 0] + ind[:, :, 1] * params['H']
            #ind = tf.expand_dims(tf.nn.top_k(ind, k=params['N'] * 2).values, axis=-1)
            #ind = tf.concat([ind % params['H'], ind // params['H']], axis=-1)

            node_index_0 = tf.cast(ind[:, :, 0] * params['W'] / (16 * 16) + ind[:, :, 1] / 16,dtype=tf.int32)
            gconv0 = tf.batch_gather(tf.reshape(features[0], [features[0].shape.as_list()[0], -1, features[0].shape.as_list()[-1]]),node_index_0)

            node_index_1 = tf.cast(ind[:, :, 0] * params['W'] / (8 * 8) + ind[:, :, 1] / 8,dtype=tf.int32)
            gconv1 = tf.batch_gather(tf.reshape(features[1], [features[1].shape.as_list()[0], -1, features[1].shape.as_list()[-1]]),node_index_1)

            node_index_2 = tf.cast(ind[:, :, 0] * params['W'] / (4 * 4) + ind[:, :, 1] / 4,dtype=tf.int32)
            gconv2 = tf.batch_gather(tf.reshape(features[2], [features[2].shape.as_list()[0], -1, features[2].shape.as_list()[-1]]),node_index_2)

            node_index_3 = tf.cast(ind[:, :, 0] * params['W'] / (2 * 2) + ind[:, :, 1] / 2,dtype=tf.int32)
            gconv3 = tf.batch_gather(tf.reshape(features[3], [features[3].shape.as_list()[0], -1, features[3].shape.as_list()[-1]]),node_index_3)

            node_index_4 = tf.cast(ind[:, :, 0] * params['W'] + ind[:, :, 1], dtype=tf.int32)
            gconv4 = tf.batch_gather(tf.reshape(features[4], [features[4].shape.as_list()[0], -1, features[4].shape.as_list()[-1]]),node_index_4)

            feature = tf.concat([gconv0, gconv1, gconv2, gconv3, gconv4, ind], axis=-1)

            # 一维卷积过程
            conv1_1 = c1rb(feature, 64, 9, is_training, scope='g_ng1_1_' + i)
            conv1_2 = c1rb(conv1_1, 64, 9, is_training, scope='g_ng1_2_' + i)
            pool1 = tf.layers.max_pooling1d(conv1_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling1')

            conv2_1 = c1rb(pool1, 128, 9, is_training, scope='g_ng2_1_' + i)
            conv2_2 = c1rb(conv2_1, 128, 9, is_training, scope='g_ng2_2_' + i)
            pool2 = tf.layers.max_pooling1d(conv2_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling2')

            conv3_1 = c1rb(pool2, 256, 9, is_training, scope='g_ng3_1_' + i)
            conv3_2 = c1rb(conv3_1, 256, 9, is_training, scope='g_ng3_2_' + i)
            pool3 = tf.layers.max_pooling1d(conv3_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling3')

            conv4_1 = c1rb(pool3, 512, 9, is_training, scope='g_ng4_1_' + i)
            conv4_2 = c1rb(conv4_1, 512, 9, is_training, scope='g_ng4_2_' + i)
            pool4 = tf.layers.max_pooling1d(conv4_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling4')

            conv5_1 = c1rb(pool4, 1024, 9, is_training, scope='g_ng5_1_' + i)
            conv5_2 = c1rb(conv5_1, 1024, 9, is_training, scope='g_ng5_2_' + i)
            up1 = _upscore_layer_1d(conv5_2, shape=conv4_2.shape.as_list(), stride=2, name='g_ng_up1', reuse=reuse,num_class=conv5_2.shape.as_list()[-1],filter=filter1)

            conv6_1 = c1rb(up1, 512, 9, is_training, scope='g_ng6_1_' + i)
            conv6_2 = c1rb(conv6_1, 512, 9, is_training, scope='g_ng6_2_' + i)
            up2 = _upscore_layer_1d(conv6_2, shape=conv3_2.shape.as_list(), stride=2, name='g_ng_up2',reuse=reuse, num_class=conv6_2.shape.as_list()[-1],filter=filter2)

            conv7_1 = c1rb(up2, 256, 9, is_training, scope='g_ng7_1_' + i)
            conv7_2 = c1rb(conv7_1, 256, 9, is_training, scope='g_ng7_2_' + i)
            up3 = _upscore_layer_1d(conv7_2, shape=conv2_2.shape.as_list(), stride=2, name='g_ng_up3',reuse=reuse, num_class=conv7_2.shape.as_list()[-1],filter=filter3)

            conv8_1 = c1rb(up3, 128, 9, is_training, scope='g_ng8_1_' + i)
            conv8_2 = c1rb(conv8_1, 64, 9, is_training, scope='g_ng8_2_' + i)
            conv8_3 = c1rb(conv8_2, 2, 9, is_training, scope='g_ng8_3_' + i)

            return conv8_3

        #N*2个节点的置信度最小值
        min_value=tf.reduce_min(node.values[batch//2,:])
        result=tf.cond(tf.less(min_value,tf.constant(params['prob'])), true_fn=true_proc,false_fn=false_proc)

        return result

def NG_multilayer_index1(ind,features,params,is_training,i,reuse):
    with tf.variable_scope('d_ng_'+i, reuse=reuse):
        # 反卷积变量的初始化
        filter1 = get_conv_filter([4, 1, 1024, 1024], stddev=5e-4, wd=0)
        filter2 = get_conv_filter([4, 1, 512, 512], stddev=5e-4, wd=0)
        filter3 = get_conv_filter([4, 1, 256, 256], stddev=5e-4, wd=0)

        node_index_0 = tf.cast(ind[:, :, 0] * params['W'] / (16 * 16) + ind[:, :, 1] / 16, dtype=tf.int32)
        gconv0 = tf.batch_gather(tf.reshape(features[0], [features[0].shape.as_list()[0], -1, features[0].shape.as_list()[-1]]),node_index_0)

        node_index_1 = tf.cast(ind[:, :, 0] * params['W'] / (8 * 8) + ind[:, :, 1] / 8, dtype=tf.int32)
        gconv1 = tf.batch_gather(tf.reshape(features[1], [features[1].shape.as_list()[0], -1, features[1].shape.as_list()[-1]]),node_index_1)

        node_index_2 = tf.cast(ind[:, :, 0] * params['W'] / (4 * 4) + ind[:, :, 1] / 4, dtype=tf.int32)
        gconv2 = tf.batch_gather(tf.reshape(features[2], [features[2].shape.as_list()[0], -1, features[2].shape.as_list()[-1]]),node_index_2)

        node_index_3 = tf.cast(ind[:, :, 0] * params['W'] / (2 * 2) + ind[:, :, 1] / 2, dtype=tf.int32)
        gconv3 = tf.batch_gather(tf.reshape(features[3], [features[3].shape.as_list()[0], -1, features[3].shape.as_list()[-1]]),node_index_3)

        node_index_4 = tf.cast(ind[:, :, 0] * params['W'] + ind[:, :, 1], dtype=tf.int32)
        gconv4 = tf.batch_gather(tf.reshape(features[4], [features[4].shape.as_list()[0], -1, features[4].shape.as_list()[-1]]),node_index_4)

        feature = tf.concat([gconv0, gconv1, gconv2, gconv3, gconv4, ind], axis=-1)

        # 一维卷积过程
        conv1_1 = c1rb(feature, 64, 9, is_training, scope='g_ng1_1_' + i)
        conv1_2 = c1rb(conv1_1, 64, 9, is_training, scope='g_ng1_2_' + i)
        pool1 = tf.layers.max_pooling1d(conv1_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling1')

        conv2_1 = c1rb(pool1, 128, 9, is_training, scope='g_ng2_1_' + i)
        conv2_2 = c1rb(conv2_1, 128, 9, is_training, scope='g_ng2_2_' + i)
        pool2 = tf.layers.max_pooling1d(conv2_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling2')

        conv3_1 = c1rb(pool2, 256, 9, is_training, scope='g_ng3_1_' + i)
        conv3_2 = c1rb(conv3_1, 256, 9, is_training, scope='g_ng3_2_' + i)
        pool3 = tf.layers.max_pooling1d(conv3_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling3')

        conv4_1 = c1rb(pool3, 512, 9, is_training, scope='g_ng4_1_' + i)
        conv4_2 = c1rb(conv4_1, 512, 9, is_training, scope='g_ng4_2_' + i)
        pool4 = tf.layers.max_pooling1d(conv4_2, pool_size=2, strides=2, padding='same', name='g_ng_pooling4')

        conv5_1 = c1rb(pool4, 1024, 9, is_training, scope='g_ng5_1_' + i)
        conv5_2 = c1rb(conv5_1, 1024, 9, is_training, scope='g_ng5_2_' + i)
        up1 = _upscore_layer_1d(conv5_2, shape=conv4_2.shape.as_list(), stride=2, name='g_ng_up1', reuse=reuse,
                                num_class=conv5_2.shape.as_list()[-1], filter=filter1)

        conv6_1 = c1rb(up1, 512, 9, is_training, scope='g_ng6_1_' + i)
        conv6_2 = c1rb(conv6_1, 512, 9, is_training, scope='g_ng6_2_' + i)
        up2 = _upscore_layer_1d(conv6_2, shape=conv3_2.shape.as_list(), stride=2, name='g_ng_up2', reuse=reuse,
                                num_class=conv6_2.shape.as_list()[-1], filter=filter2)

        conv7_1 = c1rb(up2, 256, 9, is_training, scope='g_ng7_1_' + i)
        conv7_2 = c1rb(conv7_1, 256, 9, is_training, scope='g_ng7_2_' + i)
        up3 = _upscore_layer_1d(conv7_2, shape=conv2_2.shape.as_list(), stride=2, name='g_ng_up3', reuse=reuse,
                                num_class=conv7_2.shape.as_list()[-1], filter=filter3)

        conv8_1 = c1rb(up3, 128, 9, is_training, scope='g_ng8_1_' + i)
        conv8_2 = c1rb(conv8_1, 64, 9, is_training, scope='g_ng8_2_' + i)
        conv8_3 = c1rb(conv8_2, 2, 9, is_training, scope='g_ng8_3_' + i)

        return conv8_3, tf.concat([conv2_1,conv2_2,conv8_1,conv8_2],axis=-1)

#图卷积操作模块
def CGB1(feature,is_training,params,i,reuse):
    with tf.variable_scope('d_gcn_'+str(i), reuse=reuse):
        for j in range(params['GLayers']):
            # 1.同一层中的节点融合 2.不同层节点融合
            feature1 = c1rb(feature, params['channels'][j], params['Node1'], is_training, params['T1'], scope='d_gcn1_' + str(i) + str(j))
            feature2 = c1rb(tf.transpose(feature,[1,0,2]), params['channels'][j], params['Node2'], is_training, scope='d_gcn2_' + str(i) + str(j))

            # 3.节点特征学习
            feature = tf.concat([feature1,tf.transpose(feature2,[1,0,2])],axis=-1)
            feature = c1rb(feature, params['channels'][j], params['Node1'], is_training,activation=False, scope='d_gcn3_' + str(i) + str(j))

    return feature

def CGB(feature,is_training,params,i,reuse):
    with tf.variable_scope('d_gcn_'+str(i), reuse=reuse):
        feature=tf.expand_dims(feature,axis=0)
        for j in range(params['GLayers']):
            # 1.同一层中的节点融合 2.不同层节点融合
            feature = c2rb(feature, params['channels'][j], [params['Node2'],params['Node1']], is_training, dilation_rate=(1,params['T1']), scope='d_gcn1_' + str(i) + str(j))

    return feature[0]

#用来预测边界和分割靶区,编码区的边界毫无用处,2D
def inference_seg(images, num_classes,is_training,reuse):
    with tf.variable_scope("Generator", reuse=reuse):
        with tf.variable_scope('g_pool1', reuse=reuse):
            conv1_1 = c2rb(images, 64, [3, 3], is_training, scope='g_conv1_1')
            conv1_2 = c2rb(conv1_1, 64, [3, 3], is_training, scope='g_conv1_2')
            pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling1')

        with tf.variable_scope('g_pool2', reuse=reuse):
            conv2_1 = c2rb(pool1, 128, [3, 3], is_training, scope='g_conv2_1')
            conv2_2 = c2rb(conv2_1, 128, [3, 3], is_training, scope='g_conv2_2')
            map0, bpb0 = BPB(conv2_2, is_training, '0', reuse)
            pool2 = tf.layers.max_pooling2d(bpb0, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling2')

        with tf.variable_scope('g_pool3', reuse=reuse):
            conv3_1 = c2rb(pool2, 256, [3, 3], is_training, scope='g_conv3_1')
            conv3_2 = c2rb(conv3_1, 256, [3, 3], is_training, scope='g_conv3_2')
            map1, bpb1 = BPB(conv3_2, is_training, '1',reuse)
            pool3 = tf.layers.max_pooling2d(bpb1, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling3')

        with tf.variable_scope('g_pool4', reuse=reuse):
            conv4_1 = c2rb(pool3, 512, [3, 3], is_training, scope='g_conv4_1')
            conv4_2 = c2rb(conv4_1, 512, [3, 3], is_training, scope='g_conv4_2')
            map2, bpb2 = BPB(conv4_2, is_training, '2',reuse)
            pool4 = tf.layers.max_pooling2d(bpb2, pool_size=(2, 2), strides=(2, 2), padding='same')

        with tf.variable_scope('g_vally', reuse=reuse):
            conv5_1 = c2rb(pool4, 1024, [3, 3], is_training, scope='g_conv5_1')
            conv5_2 = c2rb(conv5_1, 1024, [3, 3], is_training, scope='g_conv5_2')
            map3, bpb3 = BPB(conv5_2, is_training, '3',reuse)

            #这边做一个全连接层，用来图像分类
            fc1 = tf.layers.average_pooling2d(bpb3, pool_size=(bpb3.shape.as_list()[1], bpb3.shape.as_list()[2]),strides=(1, 1), padding='valid')
            fc2 = tf.layers.flatten(fc1)
            fc3 = tf.layers.dense(fc2, 4096, trainable=is_training)
            fc4 = tf.layers.dense(fc3, 2048, trainable=is_training)
            fc5 = tf.layers.dense(fc4, 1024, trainable=is_training)
            fc = tf.layers.dense(fc5, 1, trainable=is_training)

            # fc1 = c2rb(bpb3, 4096, [bpb3.shape.as_list()[1], bpb3.shape.as_list()[2]], is_training,padding='valid', scope='g_fc_1')
            # fc2 = c2rb(fc1, 2048, [1, 1], is_training, scope='g_fc_2')
            # fc3 = c2rb(fc2, 1024, [1, 1], is_training, scope='g_fc_3')
            # fc = c2rb(fc3, 1, [1, 1], is_training,activation=False, scope='g_fc')
            # fc=tf.reshape(fc,[fc.shape.as_list()[0]])

        with tf.variable_scope('g_up6', reuse=reuse):
            up1 = _upscore_layer(bpb3, shape=conv4_2.shape.as_list(), ksize=4, stride=2, name='g_up1',reuse=reuse,num_class=bpb3.shape.as_list()[-1])
            concat6 = tf.concat([up1, bpb2], axis=3)
            conv6_1 = c2rb(concat6, 512, [3, 3], is_training, scope='g_conv6_1')
            conv6_2 = c2rb(conv6_1, 512, [3, 3], is_training, scope='g_conv6_2')
            map4, bpb4 = BPB(conv6_2, is_training, '4',reuse)

        with tf.variable_scope('g_up7', reuse=reuse):
            up2 = _upscore_layer(bpb4, shape=conv3_2.shape.as_list(), ksize=4, stride=2, name='g_up2',reuse=reuse,num_class=bpb4.shape.as_list()[-1])
            concat7 = tf.concat([up2, bpb1], axis=3)
            conv7_1 = c2rb(concat7, 256, [3, 3], is_training, scope='g_conv7_1')
            conv7_2 = c2rb(conv7_1, 256, [3, 3], is_training, scope='g_conv7_2')
            map5, bpb5 = BPB(conv7_2, is_training, '5',reuse)

        with tf.variable_scope('g_up8', reuse=reuse):
            up3 = _upscore_layer(bpb5, shape=conv2_2.shape.as_list(), ksize=4, stride=2, name='g_up3',reuse=reuse,num_class=bpb5.shape.as_list()[-1])
            concat8 = tf.concat([up3, conv2_2], axis=3)
            conv8_1 = c2rb(concat8, 128, [3, 3], is_training, scope='g_conv8_1')
            conv8_2 = c2rb(conv8_1, 128, [3, 3], is_training, scope='g_conv8_2')
            map6, bpb6 = BPB(conv8_2, is_training, '6',reuse)

        with tf.variable_scope('g_up9', reuse=reuse):
            up4 = _upscore_layer(bpb6, shape=conv1_2.shape.as_list(), ksize=4, stride=2, name='g_up4',reuse=reuse,num_class=bpb6.shape.as_list()[-1])
            concat9 = tf.concat([up4, conv1_2], axis=3)
            conv9_1 = c2rb(concat9, 64, [3, 3], is_training, scope='g_conv9_1')
            conv9_2 = c2rb(conv9_1, 64, [3, 3], is_training, scope='g_conv9_2')
            map7, bpb7 = BPB(conv9_2, is_training, '6', reuse)

            #对图像进行一个图像分类层次的label
            #ipb = tf.add(bpb7, tf.multiply(bpb7, tf.reshape(tf.sigmoid(fc),[fc.shape.as_list()[0]])))

        with tf.variable_scope('g_final', reuse=reuse):
            conv10 = c2rb(bpb7, num_classes, [1, 1], is_training, scope='g_conv10')
            softmax_conv10 = tf.nn.softmax(conv10, dim=3, name='softmax_conv10')

    return conv10, softmax_conv10,fc,[map0,map1,map2,map3,map4,map5,map6,map7],[bpb3,bpb2,bpb1,bpb0,conv1_2]
    #return conv10, softmax_conv10,[map0,map1,map2,map3,map4,map5,map6,map7],[bpb3,bpb4,bpb5,bpb6,bpb7]

#2*2D
def inference_seg1(images, num_classes,is_training,reuse):
    with tf.variable_scope("Generator", reuse=reuse):
        with tf.variable_scope('g_pool1', reuse=reuse):
            conv1_1_0 = c2rb(images, 64, [3, 3], is_training, scope='g_conv1_1_0')
            conv1_2_0 = c2rb(conv1_1_0, 64, [3, 3], is_training, scope='g_conv1_2_0')

            #另一个维度的卷积
            conv1_1_1 = c2rb(tf.transpose(images,[2,0,1,3]), 64, [3, 3], is_training, scope='g_conv1_1_1')
            conv1_2_1 = c2rb(conv1_1_1, 64, [3, 3], is_training, scope='g_conv1_2_1')
            conv1_2_1=tf.transpose(conv1_2_1,[1,2,0,3])
            conv1_2=tf.concat([conv1_2_0,conv1_2_1],axis=-1)

            pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling1')

        with tf.variable_scope('g_pool2', reuse=reuse):
            conv2_1_0 = c2rb(pool1, 128, [3, 3], is_training, scope='g_conv2_1_0')
            conv2_2_0 = c2rb(conv2_1_0, 128, [3, 3], is_training, scope='g_conv2_2_0')

            conv2_1_1 = c2rb(tf.transpose(pool1,[2,0,1,3]), 128, [3, 3], is_training, scope='g_conv2_1_1')
            conv2_2_1 = c2rb(conv2_1_1, 128, [3, 3], is_training, scope='g_conv2_2_1')
            conv2_2_1 = tf.transpose(conv2_2_1, [1, 2, 0, 3])
            conv2_2 = tf.concat([conv2_2_0, conv2_2_1], axis=-1)

            map0, bpb0 = BPB(conv2_2, is_training, '0', reuse)
            pool2 = tf.layers.max_pooling2d(bpb0, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling2')

        with tf.variable_scope('g_pool3', reuse=reuse):
            conv3_1_0 = c2rb(pool2, 256, [3, 3], is_training, scope='g_conv3_1_0')
            conv3_2_0 = c2rb(conv3_1_0, 256, [3, 3], is_training, scope='g_conv3_2_0')

            conv3_1_1 = c2rb(tf.transpose(pool2, [2, 0, 1, 3]), 128, [3, 3], is_training, scope='g_conv3_1_1')
            conv3_2_1 = c2rb(conv3_1_1, 128, [3, 3], is_training, scope='g_conv3_2_1')
            conv3_2_1 = tf.transpose(conv3_2_1, [1, 2, 0, 3])
            conv3_2 = tf.concat([conv3_2_0, conv3_2_1], axis=-1)

            map1, bpb1 = BPB(conv3_2, is_training, '1',reuse)
            pool3 = tf.layers.max_pooling2d(bpb1, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling3')

        with tf.variable_scope('g_pool4', reuse=reuse):
            conv4_1_0 = c2rb(pool3, 512, [3, 3], is_training, scope='g_conv4_1_0')
            conv4_2_0 = c2rb(conv4_1_0, 512, [3, 3], is_training, scope='g_conv4_2_0')

            conv4_1_1 = c2rb(tf.transpose(pool3, [2, 0, 1, 3]), 128, [3, 3], is_training, scope='g_conv4_1_1')
            conv4_2_1 = c2rb(conv4_1_1, 128, [3, 3], is_training, scope='g_conv4_2_1')
            conv4_2_1 = tf.transpose(conv4_2_1, [1, 2, 0, 3])
            conv4_2 = tf.concat([conv4_2_0, conv4_2_1], axis=-1)

            map2, bpb2 = BPB(conv4_2, is_training, '2',reuse)
            pool4 = tf.layers.max_pooling2d(bpb2, pool_size=(2, 2), strides=(2, 2), padding='same')

        with tf.variable_scope('g_vally', reuse=reuse):
            conv5_1_0 = c2rb(pool4, 1024, [3, 3], is_training, scope='g_conv5_1_0')
            conv5_2_0 = c2rb(conv5_1_0, 1024, [3, 3], is_training, scope='g_conv5_2_0')

            conv5_1_1 = c2rb(tf.transpose(pool4, [2, 0, 1, 3]), 128, [3, 3], is_training, scope='g_conv5_1_1')
            conv5_2_1 = c2rb(conv5_1_1, 128, [3, 3], is_training, scope='g_conv5_2_1')
            conv5_2_1 = tf.transpose(conv5_2_1, [1, 2, 0, 3])
            conv5_2 = tf.concat([conv5_2_0, conv5_2_1], axis=-1)

            map3, bpb3 = BPB(conv5_2, is_training, '3',reuse)

            # 这边做一个全连接层，用来图像分类
            fc1 = tf.layers.average_pooling2d(bpb3, pool_size=(bpb3.shape.as_list()[1], bpb3.shape.as_list()[2]),
                                              strides=(1, 1), padding='valid')
            fc2 = tf.layers.flatten(fc1)
            fc3 = tf.layers.dense(fc2, 4096, trainable=is_training)
            fc4 = tf.layers.dense(fc3, 2048, trainable=is_training)
            fc5 = tf.layers.dense(fc4, 1024, trainable=is_training)
            fc = tf.layers.dense(fc5, 1, trainable=is_training)

        with tf.variable_scope('g_up6', reuse=reuse):
            up1 = _upscore_layer(bpb3, shape=conv4_2.shape.as_list(), ksize=4, stride=2, name='g_up1',reuse=reuse,num_class=bpb3.shape.as_list()[-1])
            concat6 = tf.concat([up1, bpb2], axis=3)
            conv6_1 = c2rb(concat6, 512, [3, 3], is_training, scope='g_conv6_1_0')
            conv6_2 = c2rb(conv6_1, 512, [3, 3], is_training, scope='g_conv6_2_0')

            map4, bpb4 = BPB(conv6_2, is_training, '4',reuse)

        with tf.variable_scope('g_up7', reuse=reuse):
            up2 = _upscore_layer(bpb4, shape=conv3_2.shape.as_list(), ksize=4, stride=2, name='g_up2',reuse=reuse,num_class=bpb4.shape.as_list()[-1])
            concat7 = tf.concat([up2, bpb1], axis=3)
            conv7_1 = c2rb(concat7, 256, [3, 3], is_training, scope='g_conv7_1_0')
            conv7_2 = c2rb(conv7_1, 256, [3, 3], is_training, scope='g_conv7_2_0')

            map5, bpb5 = BPB(conv7_2, is_training, '5',reuse)

        with tf.variable_scope('g_up8', reuse=reuse):
            up3 = _upscore_layer(bpb5, shape=conv2_2.shape.as_list(), ksize=4, stride=2, name='g_up3',reuse=reuse,num_class=bpb5.shape.as_list()[-1])
            concat8 = tf.concat([up3, conv2_2], axis=3)
            conv8_1 = c2rb(concat8, 128, [3, 3], is_training, scope='g_conv8_1_0')
            conv8_2 = c2rb(conv8_1, 128, [3, 3], is_training, scope='g_conv8_2_0')

            map6, bpb6 = BPB(conv8_2, is_training, '6',reuse)

        with tf.variable_scope('g_up9', reuse=reuse):
            up4 = _upscore_layer(bpb6, shape=conv1_2.shape.as_list(), ksize=4, stride=2, name='g_up4',reuse=reuse,num_class=bpb6.shape.as_list()[-1])
            concat9 = tf.concat([up4, conv1_2], axis=3)
            conv9_1 = c2rb(concat9, 64, [3, 3], is_training, scope='g_conv9_1_0')
            conv9_2 = c2rb(conv9_1, 64, [3, 3], is_training, scope='g_conv9_2_0')

            map7, bpb7 = BPB(conv9_2, is_training, '6', reuse)

        with tf.variable_scope('g_final', reuse=reuse):
            conv10 = c2rb(bpb7, num_classes, [1, 1], is_training, scope='g_conv10')
            softmax_conv10 = tf.nn.softmax(conv10, dim=3, name='softmax_conv10')

    return conv10, softmax_conv10,fc,[map0,map1,map2,map3,map4,map5,map6,map7],[bpb3,bpb2,bpb1,bpb0,conv1_2]

#图像分类值被用于增强
def inference_seg2(images, num_classes,is_training,reuse):
    with tf.variable_scope("Generator", reuse=reuse):
        with tf.variable_scope('g_pool1', reuse=reuse):
            conv1_1_0 = c2rb(images, 64, [3, 3], is_training, scope='g_conv1_1_0')
            conv1_2_0 = c2rb(conv1_1_0, 64, [3, 3], is_training, scope='g_conv1_2_0')

            #另一个维度的卷积
            conv1_1_1 = c2rb(tf.transpose(images,[2,0,1,3]), 64, [3, 3], is_training, scope='g_conv1_1_1')
            conv1_2_1 = c2rb(conv1_1_1, 64, [3, 3], is_training, scope='g_conv1_2_1')
            conv1_2_1=tf.transpose(conv1_2_1,[1,2,0,3])
            conv1_2=tf.concat([conv1_2_0,conv1_2_1],axis=-1)

            pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling1')

        with tf.variable_scope('g_pool2', reuse=reuse):
            conv2_1_0 = c2rb(pool1, 128, [3, 3], is_training, scope='g_conv2_1_0')
            conv2_2_0 = c2rb(conv2_1_0, 128, [3, 3], is_training, scope='g_conv2_2_0')

            conv2_1_1 = c2rb(tf.transpose(pool1,[2,0,1,3]), 128, [3, 3], is_training, scope='g_conv2_1_1')
            conv2_2_1 = c2rb(conv2_1_1, 128, [3, 3], is_training, scope='g_conv2_2_1')
            conv2_2_1 = tf.transpose(conv2_2_1, [1, 2, 0, 3])
            conv2_2 = tf.concat([conv2_2_0, conv2_2_1], axis=-1)

            map0, bpb0 = BPB(conv2_2, is_training, '0', reuse)
            pool2 = tf.layers.max_pooling2d(bpb0, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling2')

        with tf.variable_scope('g_pool3', reuse=reuse):
            conv3_1_0 = c2rb(pool2, 256, [3, 3], is_training, scope='g_conv3_1_0')
            conv3_2_0 = c2rb(conv3_1_0, 256, [3, 3], is_training, scope='g_conv3_2_0')

            conv3_1_1 = c2rb(tf.transpose(pool2, [2, 0, 1, 3]), 128, [3, 3], is_training, scope='g_conv3_1_1')
            conv3_2_1 = c2rb(conv3_1_1, 128, [3, 3], is_training, scope='g_conv3_2_1')
            conv3_2_1 = tf.transpose(conv3_2_1, [1, 2, 0, 3])
            conv3_2 = tf.concat([conv3_2_0, conv3_2_1], axis=-1)

            map1, bpb1 = BPB(conv3_2, is_training, '1',reuse)
            pool3 = tf.layers.max_pooling2d(bpb1, pool_size=(2, 2), strides=(2, 2), padding='same', name='g_pooling3')

        with tf.variable_scope('g_pool4', reuse=reuse):
            conv4_1_0 = c2rb(pool3, 512, [3, 3], is_training, scope='g_conv4_1_0')
            conv4_2_0 = c2rb(conv4_1_0, 512, [3, 3], is_training, scope='g_conv4_2_0')

            conv4_1_1 = c2rb(tf.transpose(pool3, [2, 0, 1, 3]), 128, [3, 3], is_training, scope='g_conv4_1_1')
            conv4_2_1 = c2rb(conv4_1_1, 128, [3, 3], is_training, scope='g_conv4_2_1')
            conv4_2_1 = tf.transpose(conv4_2_1, [1, 2, 0, 3])
            conv4_2 = tf.concat([conv4_2_0, conv4_2_1], axis=-1)

            map2, bpb2 = BPB(conv4_2, is_training, '2',reuse)
            pool4 = tf.layers.max_pooling2d(bpb2, pool_size=(2, 2), strides=(2, 2), padding='same')

        with tf.variable_scope('g_vally', reuse=reuse):
            conv5_1_0 = c2rb(pool4, 1024, [3, 3], is_training, scope='g_conv5_1_0')
            conv5_2_0 = c2rb(conv5_1_0, 1024, [3, 3], is_training, scope='g_conv5_2_0')

            conv5_1_1 = c2rb(tf.transpose(pool4, [2, 0, 1, 3]), 128, [3, 3], is_training, scope='g_conv5_1_1')
            conv5_2_1 = c2rb(conv5_1_1, 128, [3, 3], is_training, scope='g_conv5_2_1')
            conv5_2_1 = tf.transpose(conv5_2_1, [1, 2, 0, 3])
            conv5_2 = tf.concat([conv5_2_0, conv5_2_1], axis=-1)

            map3, bpb3 = BPB(conv5_2, is_training, '3',reuse)

            # 这边做一个全连接层，用来图像分类
            fc1 = tf.layers.average_pooling2d(bpb3, pool_size=(bpb3.shape.as_list()[1], bpb3.shape.as_list()[2]),
                                              strides=(1, 1), padding='valid')
            fc2 = tf.layers.flatten(fc1)
            fc3 = tf.layers.dense(fc2, 4096, trainable=is_training)
            fc4 = tf.layers.dense(fc3, 2048, trainable=is_training)
            fc5 = tf.layers.dense(fc4, 1024, trainable=is_training)
            fc = tf.layers.dense(fc5, 1, trainable=is_training)

            #fc扩张维度
            fc_expand_1=tf.reshape(fc,[fc.shape.as_list()[0],1,1,1])
            fc_expand_2=tf.tile(tf.sigmoid(fc_expand_1), [1, bpb3.shape.as_list()[1], bpb3.shape.as_list()[2], 1])
            fc_expand_3=c2rb(fc_expand_2,1,[1,1],is_training,scope="fc_expand")
            bpb3_1=bpb3+tf.multiply(bpb3,fc_expand_3)

        with tf.variable_scope('g_up6', reuse=reuse):
            up1 = _upscore_layer(bpb3_1, shape=conv4_2.shape.as_list(), ksize=4, stride=2, name='g_up1',reuse=reuse,num_class=bpb3_1.shape.as_list()[-1])
            concat6 = tf.concat([up1, bpb2], axis=3)
            conv6_1 = c2rb(concat6, 512, [3, 3], is_training, scope='g_conv6_1_0')
            conv6_2 = c2rb(conv6_1, 512, [3, 3], is_training, scope='g_conv6_2_0')

            #conv6_3=conv6_2+tf.multiply(conv6_2,tf.tile(fc_expand,[1,conv6_2.shape.as_list()[1],conv6_2.shape.as_list()[2],conv6_2.shape.as_list()[3]]))
            map4, bpb4 = BPB(conv6_2, is_training, '4',reuse)

        with tf.variable_scope('g_up7', reuse=reuse):
            up2 = _upscore_layer(bpb4, shape=conv3_2.shape.as_list(), ksize=4, stride=2, name='g_up2',reuse=reuse,num_class=bpb4.shape.as_list()[-1])
            concat7 = tf.concat([up2, bpb1], axis=3)
            conv7_1 = c2rb(concat7, 256, [3, 3], is_training, scope='g_conv7_1_0')
            conv7_2 = c2rb(conv7_1, 256, [3, 3], is_training, scope='g_conv7_2_0')

            #conv7_3 = conv7_2+tf.multiply(conv7_2, tf.tile(fc_expand,[1, conv7_2.shape.as_list()[1], conv7_2.shape.as_list()[2],conv7_2.shape.as_list()[3]]))
            map5, bpb5 = BPB(conv7_2, is_training, '5',reuse)

        with tf.variable_scope('g_up8', reuse=reuse):
            up3 = _upscore_layer(bpb5, shape=conv2_2.shape.as_list(), ksize=4, stride=2, name='g_up3',reuse=reuse,num_class=bpb5.shape.as_list()[-1])
            concat8 = tf.concat([up3, conv2_2], axis=3)
            conv8_1 = c2rb(concat8, 128, [3, 3], is_training, scope='g_conv8_1_0')
            conv8_2 = c2rb(conv8_1, 128, [3, 3], is_training, scope='g_conv8_2_0')

            #conv8_3 = conv8_2+tf.multiply(conv8_2, tf.tile(fc_expand,[1, conv8_2.shape.as_list()[1], conv8_2.shape.as_list()[2],conv8_2.shape.as_list()[3]]))
            map6, bpb6 = BPB(conv8_2, is_training, '6',reuse)

        with tf.variable_scope('g_up9', reuse=reuse):
            up4 = _upscore_layer(bpb6, shape=conv1_2.shape.as_list(), ksize=4, stride=2, name='g_up4',reuse=reuse,num_class=bpb6.shape.as_list()[-1])
            concat9 = tf.concat([up4, conv1_2], axis=3)
            conv9_1 = c2rb(concat9, 64, [3, 3], is_training, scope='g_conv9_1_0')
            conv9_2 = c2rb(conv9_1, 64, [3, 3], is_training, scope='g_conv9_2_0')

            #conv9_3 = conv9_2+tf.multiply(conv9_2, tf.tile(fc_expand,[1, conv9_2.shape.as_list()[1], conv9_2.shape.as_list()[2],conv9_2.shape.as_list()[3]]))
            map7, bpb7 = BPB(conv9_2, is_training, '6', reuse)

        with tf.variable_scope('g_final', reuse=reuse):
            conv10 = c2rb(bpb7, num_classes, [1, 1], is_training, scope='g_conv10')
            softmax_conv10 = tf.nn.softmax(conv10, dim=3, name='softmax_conv10')

    return conv10, softmax_conv10,fc,[map0,map1,map2,map3,map4,map5,map6,map7],[bpb3,bpb2,bpb1,bpb0,conv1_2]

#用来节点回归
def inference_node(nodes,features,params,is_training,reuse):
    result=[nodes]

    for i in range(params['GBlocks']):
        # 当前slice不存在边界
        def true_proc():
            return nodes

        # 当前slice存在边界
        def false_proc():
            # 每个图卷积模块的参数不共享
            # 1.根据节点坐标得到特征
            # 超出坐标范围的点都要裁掉,暂时还不知道应该怎么裁
            node_index_0 = tf.cast(nodes[:, :, 0] * params['W'] / (16 * 16) + nodes[:, :, 1] / 16,dtype=tf.int32)
            gconv0 = tf.batch_gather(tf.reshape(features[0], [features[0].shape.as_list()[0], -1, features[0].shape.as_list()[-1]]),node_index_0)

            node_index_1 = tf.cast(nodes[:, :, 0] * params['W'] / (8 * 8) + nodes[:, :, 1] / 8,dtype=tf.int32)
            gconv1 = tf.batch_gather(tf.reshape(features[1], [features[1].shape.as_list()[0], -1, features[1].shape.as_list()[-1]]),node_index_1)

            node_index_2 = tf.cast(nodes[:, :, 0] * params['W'] / (4 * 4) + nodes[:, :, 1] / 4,dtype=tf.int32)
            gconv2 = tf.batch_gather(tf.reshape(features[2], [features[2].shape.as_list()[0], -1, features[2].shape.as_list()[-1]]),node_index_2)

            node_index_3 = tf.cast(nodes[:, :, 0] * params['W'] / (2 * 2) + nodes[:, :, 1] / 2,dtype=tf.int32)
            gconv3 = tf.batch_gather(tf.reshape(features[3], [features[3].shape.as_list()[0], -1, features[3].shape.as_list()[-1]]),node_index_3)

            node_index_4 = tf.cast(nodes[:, :, 0] * params['W'] + nodes[:, :, 1], dtype=tf.int32)
            gconv4 = tf.batch_gather(tf.reshape(features[4], [features[4].shape.as_list()[0], -1, features[4].shape.as_list()[-1]]),node_index_4)

            x_min = tf.tile(tf.expand_dims(tf.reduce_min(nodes[:, :, :1], axis=1), axis=-1),[1, params['N'], 1])
            y_min = tf.tile(tf.expand_dims(tf.reduce_min(nodes[:, :, 1:], axis=1), axis=-1),[1, params['N'], 1])
            feature = tf.concat([gconv0, gconv1, gconv2, gconv3, gconv4, nodes[:, :, :1] - x_min,nodes[:, :, 1:] - y_min], axis=-1)

            # 根据特征计算图卷积
            offset = CGB(feature, is_training, params, i, reuse)
            final_nodes=nodes + offset

            return final_nodes

        max_value = tf.reduce_max(nodes[params['Node2'] // 2, :])
        min_value = tf.reduce_min(nodes[params['Node2'] // 2, :])
        nodes = tf.cond(tf.equal(min_value, max_value), true_fn=true_proc, false_fn=false_proc)
        result.append(nodes)

    return result

def inference_node1(nodes,features,params,is_training,reuse):
    result=[nodes]

    for i in range(params['GBlocks']):
        # 每个图卷积模块的参数不共享
        # 1.根据节点坐标得到特征
        # 超出坐标范围的点都要裁掉,暂时还不知道应该怎么裁
        node_index_0 = tf.cast(nodes[:, :, 0] * params['W'] / (16 * 16) + nodes[:, :, 1] / 16, dtype=tf.int32)
        gconv0 = tf.batch_gather(tf.reshape(features[0], [features[0].shape.as_list()[0], -1, features[0].shape.as_list()[-1]]),node_index_0)

        node_index_1 = tf.cast(nodes[:, :, 0] * params['W'] / (8 * 8) + nodes[:, :, 1] / 8, dtype=tf.int32)
        gconv1 = tf.batch_gather(tf.reshape(features[1], [features[1].shape.as_list()[0], -1, features[1].shape.as_list()[-1]]),node_index_1)

        node_index_2 = tf.cast(nodes[:, :, 0] * params['W'] / (4 * 4) + nodes[:, :, 1] / 4, dtype=tf.int32)
        gconv2 = tf.batch_gather(tf.reshape(features[2], [features[2].shape.as_list()[0], -1, features[2].shape.as_list()[-1]]),node_index_2)

        node_index_3 = tf.cast(nodes[:, :, 0] * params['W'] / (2 * 2) + nodes[:, :, 1] / 2, dtype=tf.int32)
        gconv3 = tf.batch_gather(tf.reshape(features[3], [features[3].shape.as_list()[0], -1, features[3].shape.as_list()[-1]]),node_index_3)

        node_index_4 = tf.cast(nodes[:, :, 0] * params['W'] + nodes[:, :, 1], dtype=tf.int32)
        gconv4 = tf.batch_gather(tf.reshape(features[4], [features[4].shape.as_list()[0], -1, features[4].shape.as_list()[-1]]),node_index_4)

        x_min = tf.tile(tf.expand_dims(tf.reduce_min(nodes[:, :, :1], axis=1), axis=-1), [1, params['N'], 1])
        y_min = tf.tile(tf.expand_dims(tf.reduce_min(nodes[:, :, 1:], axis=1), axis=-1), [1, params['N'], 1])
        feature = tf.concat([gconv0, gconv1, gconv2, gconv3, gconv4,nodes[:, :, :1], nodes[:, :, :1] - x_min,nodes[:, :, 1:], nodes[:, :, 1:] - y_min],axis=-1)

        # 根据特征计算图卷积
        offset = CGB(feature, is_training, params, i, reuse)
        nodes = nodes + offset
        result.append(nodes)

    return result

def inference_curve(images,params, num_classes,is_training,reuse):
    # 分割靶区和边界
    logit, softmax,fc, map, feature = inference_seg(images, num_classes, is_training, reuse)

    # 根据boundary产生初步的节点坐标
    node=NG_multilayer_index(softmax,feature, params, is_training, '0', reuse)

    #节点回归
    final_node=inference_node(node,feature,params,is_training,reuse)

    return logit,softmax,fc,map,final_node