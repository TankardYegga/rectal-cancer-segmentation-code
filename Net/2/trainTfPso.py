import math
import tensorflow as tf
from Net.model import inference_seg2,inference_node1
from NetTf.ops import loss_calc_seg,loss_calc_class,loss_calc_boundary,loss_calc_node, training_var, evaluation
from Net.preprocessing import *
from Net.Point import SelectPoint
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

for it in range(5):
    TEST=it

    # 数据集设置
    CHANNELS = 1  # 一个原始图像通道

    # 网络训练设置
    NUM_EPOCHS = 30
    BATCH_SIZE = params['Batch_size']
    MAX_STEP = NUM_EPOCHS * math.ceil(4700 / BATCH_SIZE)
    LEARNING_RATE_SEG = 1e-05
    LEARNING_RATE_NODE = 1e-04
    REGRESSION_INTERVAL = 2
    SAVE_RESULTS_INTERVAL = 2000
    SAVE_CHECKPOINT_INTERVAL = 2000
    LOG_DIR = '../logs/'
    CLASSES_NUM = 2
    CHECKPOINT_DIR = 'F:/data/targetarea/model/14/' + str(TEST) + "/"
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
            boundary[i] = tf.placeholder(tf.float32,[BATCH_SIZE, IMAGE_HIGHT // math.pow(2, i), IMAGE_WIDTH // math.pow(2, i), 1])

        location = tf.placeholder(tf.float32, [BATCH_SIZE, 1])
        ind = tf.placeholder(tf.float32, [BATCH_SIZE, params['N'], 2])
        node = tf.placeholder(tf.float32, [BATCH_SIZE, params['N'], 2])

        logits, softmax, fc, boundarys, feature = inference_seg2(images, CLASSES_NUM, is_training=True, reuse=False)
        # nodes,feature2 = NG_multilayer_index1(ind, feature, params,is_training=True, i='0',reuse=False)
        final_node = inference_node1(ind, feature, params, is_training=True, reuse=False)

        y_pred = tf.argmax(softmax, 3)
        prob = tf.sigmoid(fc)

        g_loss = loss_calc_seg(logits=logits, labels=labels, classes=CLASSES_NUM) + \
                 loss_calc_class(logits=fc, labels=labels) + \
                 loss_calc_boundary(logits=boundarys, labels=boundary)  # 分割网络损失函数

        d_loss = loss_calc_node(logits=final_node, labels=node)  # 节点回归网络损失函数

        # 训练的参数
        total_vars = tf.trainable_variables()
        g_vars = [var for var in total_vars if "g_" in var.name]
        d_vars = [var for var in total_vars if "d_" in var.name]

        g_global_step = tf.Variable(0, name='global_step', trainable=False)
        d_global_step = tf.Variable(0, name='global_step', trainable=False)
        g_train_op = training_var(loss=g_loss, learning_rate=LEARNING_RATE_SEG, global_step=g_global_step, var=g_vars)
        d_train_op = training_var(loss=d_loss, learning_rate=LEARNING_RATE_NODE, global_step=d_global_step, var=d_vars)

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

        g_global_step_value, = sess.run([g_global_step])
        print("Last trained iteration was: ", g_global_step_value)
        while True:
            if g_global_step_value >= MAX_STEP:
                print(f"Reached MAX_STEP: {MAX_STEP} at step: {g_global_step_value}")
                break
            images_batch, labels_batch, b_batch, location_batch, node_batch = train_generator.__next__()
            feed_dict = {images: images_batch, labels: labels_batch, location: location_batch}

            for j in range(SAMPLE_NUM):
                feed_dict[boundary[j]] = b_batch[j]

            if (g_global_step_value + 1) % SAVE_RESULTS_INTERVAL == 0:
                # 1.训练
                _, loss_value, accuracy_value, glboal_step_value, y_pred_value, prob_value = sess.run(
                    [g_train_op, g_loss, accuracy, g_global_step, y_pred, prob], feed_dict=feed_dict)
                print(f"TRAIN Seg Step: {g_global_step_value}\t Loss: {loss_value}\t Accuracy: {accuracy_value}")

                # 训练节点回归
                node_batch[node_batch == 0] = params['H'] // 2
                ind_batch, deter_batch = SelectPoint(y_pred_value, 10, params['N'])
                ind_batch[ind_batch == 0] = params['H'] // 2

                # 存在边界
                for n in range(params['Batch_size']):
                    prob_value_one = prob_value[n]
                    if (prob_value_one > params['prob'] and prob_value_one < 1 - params['prob']  and g_global_step_value > SAVE_RESULTS_INTERVAL):
                    #if (prob_value_one > params['prob'] and prob_value_one < 1 - params['prob'] and deter_batch == 1 and g_global_step_value > SAVE_RESULTS_INTERVAL):
                        feed_dict[ind] = ind_batch
                        feed_dict[node] = node_batch
                        for k in range(REGRESSION_INTERVAL):
                            _, loss_value, accuracy_value, d_global_step_value, summary_str = sess.run(
                                [d_train_op, d_loss, accuracy, d_global_step, summary], feed_dict=feed_dict)
                            train_writer.add_summary(summary_str, global_step=g_global_step_value)
                            print(
                                f"TRAIN Node Step: {d_global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

                        break

                # 2.验证
                images_batch, labels_batch, b_batch, location_batch, node_batch = test_generator.__next__()
                feed_dict = {images: images_batch, labels: labels_batch, location: location_batch}

                for j in range(SAMPLE_NUM):
                    feed_dict[boundary[j]] = b_batch[j]

                loss_value, accuracy_value, g_global_step_value, y_pred_value, prob_value = sess.run(
                    [g_loss, accuracy, g_global_step, y_pred, prob], feed_dict=feed_dict)
                print(f"TEST  Step: {g_global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

                node_batch[node_batch == 0] = params['H'] // 2
                ind_batch, deter_batch = SelectPoint(y_pred_value, 10, params['N'])
                ind_batch[ind_batch == 0] = params['H'] // 2

                # 存在边界
                for n in range(params['Batch_size']):
                    prob_value_one = prob_value[n]
                    if (prob_value_one > params['prob'] and prob_value_one < 1 - params['prob'] and g_global_step_value > SAVE_RESULTS_INTERVAL):
                    #if (prob_value_one > params['prob'] and prob_value_one < 1 - params['prob'] and deter_batch == 1 and g_global_step_value > SAVE_RESULTS_INTERVAL):
                        feed_dict[ind] = ind_batch
                        feed_dict[node] = node_batch
                        for k in range(REGRESSION_INTERVAL):
                            loss_value, accuracy_value, d_global_step_value, summary_str = sess.run(
                                [d_loss, accuracy, d_global_step, summary], feed_dict=feed_dict)
                            test_writer.add_summary(summary_str, global_step=g_global_step_value)
                            print(
                                f"TEST Node Step: {d_global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

                        break
            else:
                # 训练分割
                _, loss_value, accuracy_value, g_global_step_value, y_pred_value, prob_value = sess.run(
                    [g_train_op, g_loss, accuracy, g_global_step, y_pred, prob], feed_dict=feed_dict)
                print(f"TRAIN Seg Step: {g_global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

                # 训练节点回归
                node_batch[node_batch == 0] = params['H'] // 2
                ind_batch, deter_batch = SelectPoint(y_pred_value, 10, params['N'])
                ind_batch[ind_batch == 0] = params['H'] // 2

                # 存在边界
                for n in range(params['Batch_size']):
                    prob_value_one = prob_value[n]
                    if (prob_value_one > params['prob'] and prob_value_one < 1 - params['prob'] and g_global_step_value > SAVE_RESULTS_INTERVAL):
                    #if (prob_value_one > params['prob'] and prob_value_one < 1 - params['prob'] and deter_batch == 1 and g_global_step_value > SAVE_RESULTS_INTERVAL):
                        feed_dict[ind] = ind_batch
                        feed_dict[node] = node_batch
                        for k in range(REGRESSION_INTERVAL):
                            _, loss_value, accuracy_value, d_global_step_value = sess.run(
                                [d_train_op, d_loss, accuracy, d_global_step], feed_dict=feed_dict)
                            print(
                                f"TRAIN Node Step: {d_global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

                        break

            if g_global_step_value % SAVE_CHECKPOINT_INTERVAL == 0:
                saver.save(sess, CHECKPOINT_FL, global_step=g_global_step_value)
                print("Checkpoint Saved")
        # try:

        # except Exception as e:
        #     print('Exception')
        #     print(e)
