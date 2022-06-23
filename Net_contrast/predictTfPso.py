import tensorflow as tf
from Utils.util import dsc_similarity_coef,avd_similarity_coef,hd_similarity_coef
from Net_contrast.model import inference_2D_unet,inference_segnet,inference_ddcnn,inference_dd_resnet
from Net_contrast.preprocessing import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

for it in range(5):
    TEST=it

    BATCH_SIZE = 1
    CLASSES_NUM = 2
    CHANNELS = 1

    CHECKPOINT_FILE = 'F:/data/targetarea/model-contrast/unet/' + str(TEST) + "/"
    save_path = 'F:/data/targetarea/result-contrast/unet/' + str(TEST) + "/"
    pre_name = '_pre_label_ctv.nii.gz'

    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, CHANNELS])
        labels = tf.placeholder(tf.int64, [BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, 1])

        data_generator = DataGetterPso(TEST)

        test_x, test_y, depths, ids = data_generator.load_test()
        logits, softmax = inference_2D_unet(images, CLASSES_NUM)
        affines = data_generator.affines

        y_pred = tf.argmax(softmax, 3)
        y_pred = y_pred[0, :, :]
        dsc_all = 0
        avd_all = 0
        hd_all = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_FILE)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                for index in range(len(ids)):
                    depth = int(depths[index])
                    predictions = np.zeros((IMAGE_HIGHT, IMAGE_WIDTH, depth))
                    labels_example = np.zeros((IMAGE_HIGHT, IMAGE_WIDTH, depth))
                    for i in range(depth):
                        images_batch = np.expand_dims(test_x[index][:, :, i], 0)
                        labels_batch = np.expand_dims(test_y[index][:, :, i], 0)
                        feed_dict = {images: images_batch, labels: labels_batch}
                        y_pred_value = sess.run(y_pred, feed_dict=feed_dict)
                        predictions[:, :, i] = y_pred_value
                        labels_example[:, :, i] = np.squeeze(labels_batch, -1)

                    dsc = dsc_similarity_coef(predictions, labels_example, False, CLASSES_NUM)
                    dsc = np.asarray(dsc)
                    dsc_all = dsc_all + dsc

                    avd = avd_similarity_coef(predictions, labels_example, False, CLASSES_NUM)
                    avd = np.asarray(avd)
                    avd_all = avd_all + avd

                    hd = hd_similarity_coef(predictions, labels_example, False, CLASSES_NUM)
                    hd = np.asarray(hd)
                    hd_all = hd_all + hd

                    # print(str(ids[index]) + ":" + str(dsc))
                    print(str(ids[index]) + ":  dsc:" + str(dsc) + "   avd:" + str(avd) + "   hd:" + str(hd))

                    # save result
                    save_pred = np.zeros([512, 512, depth])
                    save_pred[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :] = predictions
                    save_nii = nib.Nifti1Image(save_pred, affines[index])
                    nib.save(save_nii, save_path + str(ids[index]) + pre_name)

                # print(str(dsc_all / len(ids)))
                print("dsc:" + str(dsc_all / len(ids)) + "   avd:" + str(avd_all / len(ids)) + "   hd:" + str(
                    hd_all / len(ids)))
            else:
                print('No checkpoint file found!')
