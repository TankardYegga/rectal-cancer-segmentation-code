import tensorflow as tf

#分割和判别器的损失函数
def loss_calc(logits, labels,score_fake,classes):

    class_inc_bg = classes

    labels = labels[...,0]

    # class_weights = tf.constant([[10.0/90, 10.0]])

    onehot_labels = tf.one_hot(labels, class_inc_bg)

    # weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)

    # weighted_losses = unweighted_losses * weights

    loss_1 = tf.reduce_mean(unweighted_losses)
    loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score_fake, labels=tf.ones_like(score_fake)))
    loss=loss_1+loss_2
    tf.summary.scalar('loss', loss)
    return loss

#不同类别的损失函数权重不同
def loss_calc2(logits, labels,classes):

    class_inc_bg = classes

    labels = labels[...,0]

    class_weights = tf.constant(value=[1.0,1.0,3.0,1.0,1.0],dtype=tf.float32)
    onehot_labels = tf.one_hot(labels, class_inc_bg)
    weights = tf.reduce_sum(class_weights * onehot_labels,axis=-1)

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)

    weighted_losses = unweighted_losses * weights

    loss = tf.reduce_mean(weighted_losses)
    tf.summary.scalar('loss', loss)
    return loss

def loss_calc_two_task(logits, labels,logits_0, labels_0,logits_1, labels_1,logits_2, labels_2,logits_3, labels_3,classes):
    class_inc_bg = classes
    labels = labels[...,0]
    onehot_labels = tf.one_hot(labels, class_inc_bg)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    loss = tf.reduce_mean(unweighted_losses)

    labels_0 = labels_0[..., 0]
    onehot_labels_0 = tf.one_hot(labels_0, class_inc_bg)
    unweighted_losses_0 = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels_0, logits=logits_0)
    loss_0 = tf.reduce_mean(unweighted_losses_0)

    labels_1 = labels_1[..., 0]
    onehot_labels_1 = tf.one_hot(labels_1, class_inc_bg)
    unweighted_losses_1 = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels_1, logits=logits_1)
    loss_1 = tf.reduce_mean(unweighted_losses_1)

    labels_2 = labels_2[..., 0]
    onehot_labels_2 = tf.one_hot(labels_2, class_inc_bg)
    unweighted_losses_2 = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels_2, logits=logits_2)
    loss_2 = tf.reduce_mean(unweighted_losses_2)

    labels_3 = labels_3[..., 0]
    onehot_labels_3 = tf.one_hot(labels_3, class_inc_bg)
    unweighted_losses_3 = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels_3, logits=logits_3)
    loss_3 = tf.reduce_mean(unweighted_losses_3)

    loss_all=loss+loss_0+loss_1+loss_2+loss_3
    tf.summary.scalar('loss', loss_all)
    return loss_all

#分割和边界以及判别器的损失函数
def loss_calc_bps(logits, labels,logits_1, labels_1,logits_2, labels_2,logits_3, labels_3,
                  logits_4, labels_4,logits_5, labels_5,logits_6, labels_6,score_fake,classes):
    #1.靶区分割结果
    class_inc_bg = classes
    labels = labels[...,0]
    onehot_labels = tf.one_hot(labels, class_inc_bg)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    loss = tf.reduce_mean(unweighted_losses)

    #关键点的预测结果
    pos_weight=100
    loss_1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_1,targets=labels_1,pos_weight=pos_weight))
    loss_2 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_2, targets=labels_2, pos_weight=pos_weight))
    loss_3 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_3, targets=labels_3, pos_weight=pos_weight))
    loss_4 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_4, targets=labels_4, pos_weight=pos_weight))
    loss_5 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_5, targets=labels_5, pos_weight=pos_weight))
    loss_6 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_6, targets=labels_6, pos_weight=pos_weight))

    #骗过判别器
    loss_7=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score_fake, labels=tf.ones_like(score_fake)))
    loss_all = loss+0.01*loss_1+0.01*loss_2+0.01*loss_3+0.01*loss_4+0.01*loss_5+0.01*loss_6+loss_7
    tf.summary.scalar('loss1', loss_all)
    return loss_all


#分割和边界的损失函数
def loss_calc_bps_seg(logits, labels,logits_1, labels_1,logits_2, labels_2,logits_3, labels_3,
                  logits_4, labels_4,logits_5, labels_5,logits_6, labels_6,classes):
    #1.靶区分割结果
    class_inc_bg = classes
    labels = labels[...,0]
    onehot_labels = tf.one_hot(labels, class_inc_bg)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    loss = tf.reduce_mean(unweighted_losses)

    #关键点的预测结果
    pos_weight=100
    loss_1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_1,targets=labels_1,pos_weight=pos_weight))
    loss_2 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_2, targets=labels_2, pos_weight=pos_weight))
    loss_3 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_3, targets=labels_3, pos_weight=pos_weight))
    loss_4 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_4, targets=labels_4, pos_weight=pos_weight))
    loss_5 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_5, targets=labels_5, pos_weight=pos_weight))
    loss_6 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_6, targets=labels_6, pos_weight=pos_weight))

    loss_all = loss+0.01*loss_1+0.01*loss_2+0.01*loss_3+0.01*loss_4+0.01*loss_5+0.01*loss_6
    #loss_all = loss
    tf.summary.scalar('loss1', loss_all)
    return loss_all

#分割的损失函数
def loss_calc_seg(logits, labels,classes):
    class_inc_bg = classes
    label = labels[..., 0]
    # class_weights = tf.constant([[10.0/90, 10.0]])
    onehot_labels = tf.one_hot(label, class_inc_bg)
    # weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    # weighted_losses = unweighted_losses * weights
    loss = tf.reduce_mean(unweighted_losses)
    tf.summary.scalar('loss_seg', loss)
    return loss

#图像分类损失函数
def loss_calc_class(logits, labels):
    labels_image=tf.reshape(labels,[labels.shape.as_list()[0],-1,1])
    labels_image=tf.reduce_max(labels_image,axis=1)
    labels_image=tf.cast(labels_image,dtype=tf.float32)
    unweighted_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_image, logits=logits)
    # weighted_losses = unweighted_losses * weights
    loss = tf.reduce_mean(unweighted_losses)
    tf.summary.scalar('loss_seg', loss)
    return loss

#边界的损失函数
def loss_calc_boundary(logits, labels):
    loss=0
    # pos_weight = 10
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[0], targets=labels[1], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[1], targets=labels[2], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[2], targets=labels[3], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[3], targets=labels[4], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[4], targets=labels[3], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[5], targets=labels[2], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[6], targets=labels[1], pos_weight=pos_weight))
    # loss = loss + tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits[7], targets=labels[0], pos_weight=pos_weight))


    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[0], labels=labels[1]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[1], labels=labels[2]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[2], labels=labels[3]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[3], labels=labels[4]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[4], labels=labels[3]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[5], labels=labels[2]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[6], labels=labels[1]))
    loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[7], labels=labels[0]))
    tf.summary.scalar('loss_boundary', loss)
    return loss

#节点回归的均方根误差
def loss_calc_node(logits,labels):
    loss=0
    for j in range(len(logits)):
        #loss = loss + tf.losses.mean_squared_error(labels, logits[j])
        loss = loss + tf.losses.huber_loss(labels, logits[j],delta=1.35)
    tf.summary.scalar('loss_node', loss)
    return loss

#生成器的损失函数
def loss_calc_gen(score_fake):
    # 骗过判别器
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score_fake, labels=tf.ones_like(score_fake)))
    tf.summary.scalar('loss_gen', loss)
    return loss

#判别器训练时的损失函数
def loss_calc_sbe(score_fake,score_true):
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score_true, labels=tf.ones_like(score_true)))+\
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score_fake, labels=tf.zeros_like(score_fake)))
    tf.summary.scalar('loss_sbe', loss)
    return loss

def evaluation(logits, labels,name='accuracy'):
    labels = labels[..., 0]
    correct_prediction = tf.equal(tf.argmax(logits, -1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar(name, accuracy)
    return accuracy

def training(loss, learning_rate, global_step):
    #This motif is needed to hook up the batch_norm updates to the training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op

def training_var(loss, learning_rate, global_step,var):
    #This motif is needed to hook up the batch_norm updates to the training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step,var_list=var)
    return train_op