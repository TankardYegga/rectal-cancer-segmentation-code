import tensorflow as tf
import numpy as np
from NetTf import inference
import nibabel as nib

BATCH_SIZE = 1
CLASSES_NUM = 2
IMAGE_HIGHT_MIN = 80
IMAGE_WIDTH_MIN = 75
IMAGE_HIGHT_MAX = 400
IMAGE_WIDTH_MAX = 331
IMAGE_HIGHT = IMAGE_HIGHT_MAX-IMAGE_HIGHT_MIN
IMAGE_WIDTH = IMAGE_WIDTH_MAX-IMAGE_WIDTH_MIN
CHANNELS = 1

def seg_musles(data_path,data_nii):
    CHECKPOINT_FILE = 'F:/data/pansData/radiomic/model/'
    for i in range(31,66):
        one_img_path = data_path + str(i) + data_nii
        img = nib.load(one_img_path).get_data()

        with tf.Graph().as_default() as g:
            images = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, CHANNELS])

            test_x = img[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            test_x = np.expand_dims(test_x, -1)
            print(test_x.shape)

            _, softmax = inference.inference(images, CLASSES_NUM)

            y_pred = tf.argmax(softmax, 3)
            y_pred = y_pred[0, :, :]

            saver = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(CHECKPOINT_FILE)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    depth = int(np.shape(img)[-1])
                    predictions = np.zeros((IMAGE_HIGHT, IMAGE_WIDTH, depth))
                    for j in range(depth):
                        images_batch = np.expand_dims(test_x[:, :, j], 0)
                        feed_dict = {images: images_batch}
                        y_pred_value = sess.run(y_pred, feed_dict=feed_dict)
                        predictions[:, :, j] = y_pred_value
                    print(predictions.shape)


                    # save result
                    save_pred = np.zeros([512, 512, depth])
                    save_pred[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :] = predictions
                    save_nii = nib.Nifti1Image(save_pred, nib.load(one_img_path).affine)
                    nib.save(save_nii, data_path + str(i)+"/label_psoas_major.nii.gz")
                    # print('dsc_all    :{:.5f}'.format(dsc_all / len(ids)))

                else:
                    print('No checkpoint file found!')

if __name__=='__main__':
    data_path = 'F:/data/targetarea/dataset/'
    data_nii = '/ct.nii.gz'
    seg_musles(data_path=data_path, data_nii=data_nii)






