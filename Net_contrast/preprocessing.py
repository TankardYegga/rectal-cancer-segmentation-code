import numpy as np
import nibabel as nib

"""
数据预处理模块
"""
#对每个样本进性归一化，样本可能只是一个slice，也可能是3d volume
def normalize(data):
    min = np.min(data)
    max = np.max(data)
    if not min == max:
        return min,max,(data - min)/(max - min)
    else:
        return min,max,data*0

# 直方图均衡
def image_histogram_equalization(image, number_bins=128):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

#对图像每一个slice进行直方图均衡
def histogram_equalization(img):
    for i in range(img.shape[2]):
        img_slice=img[:,:,i]
        img_slice=image_histogram_equalization(img_slice)[0]
        img[:,:,i]=img_slice
    return img

IMAGE_HIGHT_MIN = 170
IMAGE_WIDTH_MIN = 75
IMAGE_HIGHT_MAX = 490
IMAGE_WIDTH_MAX = 427
IMAGE_HIGHT = IMAGE_HIGHT_MAX-IMAGE_HIGHT_MIN
IMAGE_WIDTH = IMAGE_WIDTH_MAX-IMAGE_WIDTH_MIN

NUM = 65

# DataGetter for target area
class DataGetterPso:
    def __init__(self,test):
        self.main_dir = 'F:/data/targetarea/dataset/'
        self.img_dir='/ct.nii.gz'
        self.label_dir = ['/label_ctv.nii.gz']

        self.TEST = test
        self.ONE_PART = int(NUM * 0.2)
        self.TEST_PART = self.TEST * self.ONE_PART + 1

    def load_train(self):
        train_imgs = []
        train_labels = []
        self.affines = []
        self.depths = []
        for i in range(1, NUM+1):
            if i == 8 or (i>=self.TEST_PART and i<self.TEST_PART+self.ONE_PART):
                continue
            print('Processing Sample {}'.format(i))

            brain_dir = self.main_dir + str(i) + self.img_dir
            brain = nib.load(brain_dir).get_data()

            truth = np.zeros(np.shape(brain))
            for j in range(len(self.label_dir)):
                truth_dir = self.main_dir + str(i) + self.label_dir[j]
                truth = truth + (nib.load(truth_dir).get_data()) * (j + 1)
                truth[truth > (j + 1)] = j + 1
            affine = nib.load(brain_dir).affine
            self.affines.append(affine)
            depth = np.shape(brain)[-1]
            brain = brain[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            truth = truth[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            # 使用全部的slice
            brain = histogram_equalization(brain)  #数据预处理
            for d in range(depth):
                train_imgs.append(brain[:,:, d])
                train_labels.append(truth[:,:, d])
        train_imgs = np.asarray(train_imgs)
        train_labels = np.asarray(train_labels, dtype=np.int32)

        self.train_imgs = np.reshape(train_imgs, [len(train_imgs), IMAGE_HIGHT, IMAGE_WIDTH])
        self.train_labels = np.reshape(train_labels, [len(train_labels), IMAGE_HIGHT, IMAGE_WIDTH])
        # 增加一个维度
        self.train_imgs = np.expand_dims(self.train_imgs, axis=-1)
        self.train_labels = np.expand_dims(self.train_labels, axis=-1)
        print(np.shape(self.train_imgs))
        print(np.shape(self.train_labels))

    def load_valid(self):
        test_imgs = []
        test_labels = []

        for i in range(self.TEST_PART, self.TEST_PART+self.ONE_PART):
            if i == 8 :
                continue
            print('Processing Sample {}'.format(i))
            brain_dir = self.main_dir + str(i) + self.img_dir
            brain = nib.load(brain_dir).get_data()

            truth = np.zeros(np.shape(brain))
            for j in range(len(self.label_dir)):
                truth_dir = self.main_dir + str(i) + self.label_dir[j]
                truth = truth + (nib.load(truth_dir).get_data()) * (j + 1)
                truth[truth > (j + 1)] = j + 1
            affine = nib.load(brain_dir).affine
            depth = np.shape(brain)[-1]

            self.affines.append(affine)
            brain = brain[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            truth=truth[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            brain = histogram_equalization(brain)  # 数据预处理
            for d in range(depth):
                test_imgs.append(brain[:, :, d])
                test_labels.append(truth[:, :, d])
        test_imgs = np.asarray(test_imgs)
        test_labels = np.asarray(test_labels, dtype=np.int32)
        self.test_imgs = np.reshape(test_imgs, [len(test_imgs), IMAGE_HIGHT, IMAGE_WIDTH])
        self.test_labels = np.reshape(test_labels, [len(test_labels), IMAGE_HIGHT, IMAGE_WIDTH])

        # 增加一个维度
        self.test_imgs = np.expand_dims(self.test_imgs, axis=-1)
        self.test_labels = np.expand_dims(self.test_labels, axis=-1)

    def load_test(self):
        test_imgs = []
        test_labels = []
        self.affines = []
        ds = []
        ids=[]

        for i in range(self.TEST_PART, self.TEST_PART+self.ONE_PART):
            if (i==8):
                continue
            print('Processing Sample {}'.format(i))
            brain_dir = self.main_dir + str(i) + self.img_dir
            brain = nib.load(brain_dir).get_data()

            truth = np.zeros(np.shape(brain))
            for j in range(len(self.label_dir)):
                truth_dir = self.main_dir + str(i) + self.label_dir[j]
                truth = truth + (nib.load(truth_dir).get_data()) * (j + 1)
                truth[truth > (j + 1)] = j + 1
            affine = nib.load(brain_dir).affine
            self.affines.append(affine)
            brain=brain[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX,:]
            truth=truth[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX,:]
            ds.append(np.shape(brain)[-1])
            ids.append(i)
            brain = histogram_equalization(brain)  # 数据预处理
            brain = np.expand_dims(brain, -1)
            truth = np.expand_dims(truth, -1)
            test_imgs.append(brain)
            test_labels.append(truth)
        return test_imgs, test_labels, ds,ids

    def get_train_batch(self, batch_size):
        length = len(self.train_imgs)
        while True:
            order = np.random.permutation(length)
            for i in range(length // batch_size):
                batch_x = self.train_imgs[order[i * batch_size: i * batch_size + batch_size]]
                batch_y = self.train_labels[order[i * batch_size: i * batch_size + batch_size]]
                yield batch_x, batch_y

    # 测试时，使用这个函数，不会打乱顺序
    def get_test_batch(self, batch_size):
        length = len(self.test_imgs)
        while True:
            for i in range(length // batch_size):
                batch_x = self.test_imgs[i * batch_size: i * batch_size + batch_size]
                batch_y = self.test_labels[i * batch_size: i * batch_size + batch_size]
                yield batch_x, batch_y

if __name__ == '__main__':
    data_getter = DataGetterPso()
    test_images, test_labels = data_getter.get_train_batch()
    print(np.shape(test_images[0]))
    print(np.shape(test_labels[0]))

