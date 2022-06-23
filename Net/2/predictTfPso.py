import tensorflow as tf
from Utils.util import dsc_similarity_coef,avd_similarity_coef,hd_similarity_coef,ppv_similarity_coef
from Net.model import inference_seg2,inference_node1
from Net.preprocessing import *
from Net.Point import SelectPoint
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

for it in range(1,2):
    TEST=it

    BATCH_SIZE = params['Node2']
    CLASSES_NUM = 2
    CHANNELS = 1

    CHECKPOINT_FILE = 'F:/data/targetarea/model/12/' + str(TEST) + "/"
    save_path = 'F:/data/targetarea/result/12/' + str(TEST) + "/"
    image_name = '_ct.nii.gz'
    gt_name = '_label_ctv.nii.gz'
    pre_name = '_pre_label_ctv.nii.gz'
    cate_name = '_pre_cate.nii.gz'
    boundary_name = '_label_ctv_boundary_7.nii.gz'
    node_name = '_pre_node_0.nii.gz'

    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, CHANNELS])
        location = tf.placeholder(tf.float32, [BATCH_SIZE, 1])
        ind = tf.placeholder(tf.float32, [BATCH_SIZE, params['N'], 2])

        data_generator = DataGetterPso(TEST)

        test_x, test_y, depths, ids, test_boundary, test_location = data_generator.load_test()
        _, softmax, fc, boundarys, feature = inference_seg2(images, CLASSES_NUM, is_training=False, reuse=False)
        final_node = inference_node1(ind, feature, params, is_training=False, reuse=False)
        affines = data_generator.affines

        y_pred_ind = tf.argmax(softmax, 3)

        y_pred = tf.argmax(softmax, 3)[params['Node2'] // 2, :, :]
        cate_pred = tf.sigmoid(fc)[params['Node2'] // 2, 0]
        boundary_pred = tf.sigmoid(boundarys[7])[params['Node2'] // 2, :, :, 0]
        node_pred = final_node[0][params['Node2'] // 2, :, :]

        dsc_all = 0
        dsc_all_post = 0

        avd_all = 0
        hd_all = 0
        ppv_all=0

        dsc_all_b_0 = 0
        dsc_all_b_1 = 0
        dsc_all_b_2 = 0
        dsc_all_b_3 = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_FILE)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                for index in range(len(ids)):
                    depth = int(depths[index])
                    predictions = np.zeros((IMAGE_HIGHT, IMAGE_WIDTH, depth))
                    prediction_c = np.zeros((1, 1, depth))
                    prediction_b = np.zeros((IMAGE_HIGHT // 1, IMAGE_WIDTH // 1, depth))
                    prediction_node = np.zeros((IMAGE_HIGHT, IMAGE_WIDTH, depth))
                    for i in range(depth):
                        if (i + params['Node2'] // 2 * params['T2'] < depth and i - params['Node2'] // 2 * params[
                            'T2'] >= 0):
                            images_batch = []
                            location_batch = []
                            for k in range(params['Node2']):
                                # print(str(i) + "   " + str(i-params['Node2']//2*params['T2']+params['T2']*k))
                                images_batch.append(
                                    test_x[index][:, :, i - params['Node2'] // 2 * params['T2'] + params['T2'] * k])
                                location_batch.append(
                                    test_location[index][i - params['Node2'] // 2 * params['T2'] + params['T2'] * k])
                            images_batch = np.asarray(images_batch)
                            location_batch = np.asarray(location_batch)

                            feed_dict = {images: images_batch}
                            y_pred_ind_value = sess.run(y_pred_ind, feed_dict=feed_dict)
                            ind_batch, deter_batch = SelectPoint(y_pred_ind_value, 10, params['N'])
                            ind_batch[ind_batch == 0] = params['H'] // 2

                            feed_dict = {images: images_batch, location: location_batch, ind: ind_batch}
                            y_pred_value, cate_pred_value, boundary_pred_value = sess.run(
                                [y_pred, cate_pred, boundary_pred], feed_dict=feed_dict)
                            predictions[:, :, i] = y_pred_value
                            prediction_c[:, :, i] = cate_pred_value
                            prediction_b[:, :, i] = boundary_pred_value

                            if (cate_pred_value > params['prob'] and cate_pred_value < 1 - params['prob']):
                                node_value = sess.run(node_pred, feed_dict=feed_dict)
                                node_value[node_value < 0] = 0
                                node_value = np.concatenate((node_value[:, :1], node_value[:, 1:]), axis=-1).astype(
                                    np.int16)
                                temp = np.zeros((IMAGE_HIGHT, IMAGE_WIDTH))
                                index_node = tuple(node_value.transpose((1, 0)))
                                # temp[index_node] = 1

                                # for j in range(node_value.shape[0]):
                                #    temp[node_value[j][0],node_value[j][1]]=j+1
                                prediction_node[:, :, i] = temp

                    dsc = dsc_similarity_coef(predictions, test_y[index], False, CLASSES_NUM)
                    dsc = np.asarray(dsc)
                    dsc_all = dsc_all + dsc

                    avd = avd_similarity_coef(predictions, test_y[index], False, CLASSES_NUM)
                    avd = np.asarray(avd)
                    avd_all = avd_all + avd

                    hd = hd_similarity_coef(predictions, test_y[index][:, :, :, 0], False, CLASSES_NUM)
                    hd = np.asarray(hd)
                    hd_all = hd_all + hd

                    ppv = ppv_similarity_coef(predictions, test_y[index][:, :, :, 0], False, CLASSES_NUM)
                    ppv = np.asarray(ppv)
                    ppv_all = ppv_all + ppv

                    # print(str(ids[index]) + ":" + str(dsc))
                    print(str(ids[index]) + ":  dsc:" + str(dsc) + "   avd:" + str(avd) + "   hd:" + str(hd)+ "   ppv:" + str(ppv))

                    # save result
                    save_nii = nib.Nifti1Image(test_x[index], affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + image_name)

                    save_nii = nib.Nifti1Image(test_y[index], affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + gt_name)

                    save_nii = nib.Nifti1Image(predictions, affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + pre_name)

                    # predictions=postprocess(predictions,prediction_c)
                    # dsc = dsc_similarity_coef(predictions, test_y[index], False, CLASSES_NUM)
                    # dsc = np.asarray(dsc)
                    # dsc_all_post = dsc_all_post + dsc
                    #
                    # print(str(ids[index]) + "_postprocess:" + str(dsc))
                    # save_nii = nib.Nifti1Image(predictions, affines[index])
                    # nib.save(save_nii, save_path + str(ids[index]) +"_post"+ pre_name)

                    save_nii = nib.Nifti1Image(prediction_c, affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + cate_name)

                    save_nii = nib.Nifti1Image(prediction_b, affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + boundary_name)

                    save_nii = nib.Nifti1Image(prediction_node, affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + node_name)

                # print(str(dsc_all / len(ids)))
                print("dsc:" + str(dsc_all / len(ids)) + "   avd:" + str(avd_all / len(ids)) + "   hd:" + str(hd_all / len(ids))+ "   ppv:" + str(ppv_all / len(ids)))
                print(str(dsc_all_post / len(ids)))
            else:
                print('No checkpoint file found!')








