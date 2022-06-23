import math
import tensorflow as tf
from Net.model import inference_seg
from NetTf.ops import loss_calc_seg,loss_calc_class,loss_calc_boundary, training_var, evaluation
from Net.preprocessing import *
import os
os.environ

#5折
for it in range(1,5):
    TEST=it

    # 数据集设置
    CHANNELS = 1  # 一个原始图像通道

    # 网络训练设置
    NUM_EPOCHS = 35
    BATCH_SIZE = params['Node2']
    MAX_STEP = NUM_EPOCHS * math.ceil(4700 / BATCH_SIZE)
    LEARNING_RATE = 1e-05
    SAVE_RESULTS_INTERVAL = 2000
    SAVE_CHECKPOINT_INTERVAL = 2000
    GENERATOR_INTERVAL = 4
    LOG_DIR = '../logs/'
    CLASSES_NUM = 2
    CHECKPOINT_DIR = 'F:/data/targetarea/model/1/' + str(TEST) + "/"
    CHECKPOINT_FL = f'{CHECKPOINT_DIR}model.ckpt'
    CHECKPOINT_FILE = CHECKPOINT_DIR
    TRAIN_WRITER_DIR = f'{CHECKPOINT_DIR}/train'
    TEST_WRITER_DIR = f'{CHECKPOINT_DIR}/test'

    # 训练与测试数据生成器
    data_generator = DataGetterPso(TEST)
    data_generator.load_train()
    data_generator.load_valid()
    # 用后10个训练，前10个测试
    train_generator = data_generator.get_train_batch(BATCH_SIZE)
    test_generator = data_generator.get_test_batch(BATCH_SIZE)

    g = tf.Graph()

    with g.as_default():
        # 定义图的操作
        images = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, CHANNELS])
        labels = tf.placeholder(tf.int64, [BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, 1])
        boundary = [[] * SAMPLE_NUM for _ in range(SAMPLE_NUM)]

        for i in range(SAMPLE_NUM):
            boundary[i] = tf.placeholder(tf.float32,
                                         [BATCH_SIZE, IMAGE_HIGHT // math.pow(2, i), IMAGE_WIDTH // math.pow(2, i), 1])

        # 生成网络和判别网络
        logits, _, fc, boundarys = inference_seg(images, CLASSES_NUM, is_training=True, reuse=False)  # 分割网络
        g_loss = loss_calc_seg(logits=logits, labels=labels, classes=CLASSES_NUM) + \
                 loss_calc_class(logits=fc, labels=labels) + \
                 loss_calc_boundary(logits=boundarys, labels=boundary)

        # 训练的参数
        total_vars = tf.trainable_variables()
        g_vars = [var for var in total_vars if "g_" in var.name]

        global_step = tf.Variable(0, name='global_step', trainable=False)
        g_train_op = training(loss=g_loss, learning_rate=LEARNING_RATE, global_step=global_step, var=g_vars)

        accuracy = evaluation(logits=logits, labels=labels)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())

    sm = tf.train.SessionManager(graph=g)

    with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:
        sess.run(init)
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_FILE)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)
        test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

        global_step_value, = sess.run([global_step])
        print("Last trained iteration was: ", global_step_value)
        while True:
            if global_step_value >= MAX_STEP:
                print(f"Reached MAX_STEP: {MAX_STEP} at step: {global_step_value}")
                break
            images_batch, labels_batch, b_batch, location_batch, node_batch = train_generator.__next__()

            feed_dict = {images: images_batch, labels: labels_batch}

            for j in range(SAMPLE_NUM):
                feed_dict[boundary[j]] = b_batch[j]

            if (global_step_value + 1) % SAVE_RESULTS_INTERVAL == 0:
                _, loss_value, accuracy_value, glboal_step_value, summary_str = \
                    sess.run([g_train_op, g_loss, accuracy, global_step, summary], feed_dict=feed_dict)
                train_writer.add_summary(summary_str, global_step=global_step_value)
                print(f"TRAIN Generator Step: {global_step_value}\t Loss: {loss_value}\t Accuracy: {accuracy_value}")

                images_batch, labels_batch, b_batch, location_batch, node_batch = test_generator.__next__()

                feed_dict = {images: images_batch, labels: labels_batch}

                for j in range(SAMPLE_NUM):
                    feed_dict[boundary[j]] = b_batch[j]

                loss_value, accuracy_value, global_step_value, summary_str = \
                    sess.run([g_loss, accuracy, global_step, summary], feed_dict=feed_dict)
                test_writer.add_summary(summary_str, global_step=global_step_value)
                print(f"TEST  Step: {global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")
            else:
                # 训练生成器
                _, loss_value, accuracy_value, global_step_value = sess.run([g_train_op, g_loss, accuracy, global_step],
                                                                            feed_dict=feed_dict)
                print(f"TRAIN Generator Step: {global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")
            if global_step_value % SAVE_CHECKPOINT_INTERVAL == 0:
                saver.save(sess, CHECKPOINT_FL, global_step=global_step_value)
                print("Checkpoint Saved")
        # try:

        # except Exception as e:
        #     print('Exception')
        #     print(e)
