import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl

IMAGE_HIGHT_MIN = 90
IMAGE_WIDTH_MIN = 4
IMAGE_HIGHT_MAX = 298
IMAGE_WIDTH_MAX = 196
IMAGE_HIGHT = IMAGE_HIGHT_MAX-IMAGE_HIGHT_MIN
IMAGE_WIDTH = IMAGE_WIDTH_MAX-IMAGE_WIDTH_MIN

SAMPLE_NUM=5  #多尺度边界图

NUM = 65

#后期实验需要变的参数
#params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'GBlocks':5,'GLayers':5,'N':128,'T1':2,'T2':2,'Node1':9,'Node2':8,'prob':0.9,'coor':10,'channels' : [128,128, 128, 64, 2]} #实验1，3
#params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'GBlocks':5,'GLayers':5,'N':128,'T1':1,'T2':1,'Node1':9,'Node2':5,'prob':0.3,'coor':10,'channels' : [128,128, 128, 64, 2]} #实验2
#params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'GBlocks':5,'GLayers':5,'N':128,'T1':1,'T2':1,'Node1':9,'Node2':5,'prob':0.1,'coor':10,'channels' : [128,128, 128, 64, 2]} #实验5
#params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'GBlocks':5,'GLayers':5,'N':192,'T1':1,'T2':1,'Node1':9,'Node2':5,'prob':0.3,'coor':10,'channels' : [128,128, 128, 64, 2]} #实验6(64),7(32),8(128),9(64)，10(32),11(192)
#params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'GBlocks':5,'GLayers':5,'N':128,'T1':1,'T2':1,'Node1':9,'Node2':5,'prob':0.1,'coor':10,'channels' : [128,128, 128, 64, 2]} #实验12
#params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'GBlocks':5,'GLayers':5,'N':32,'T1':1,'T2':1,'Node1':9,'Node2':5,'prob':0.05,'coor':10,'channels' : [128,128, 128, 64, 2]} #实验13
params={'H':IMAGE_HIGHT,'W':IMAGE_WIDTH,'Batch_size':6,'GBlocks':3,'GLayers':3,'N':128,'T1':2,'T2':2,'Node1':9,'Node2':3,'prob':0.1,'channels' : [128,64, 2]} #实验14

# DataGetter for target area
class DataGetterPso:
    def __init__(self,test):
        self.main_dir = 'F:/data/targetarea/dataset/'
        self.img_dir='/ct_crop.nii.gz'
        self.label_dir = ['/label_ctv_crop.nii.gz']
        self.boundary_dir = ['/label_bone_boundary_','/label_psoas_major_boundary_','/label_ctv_boundary_']  # 0:下采样0次，1：下采样一次，2：下采样2次的边界
        self.node_dir='/label_point_crop_40000_'+str(params['N'])+'.npy'  #样本边界上的节点

        self.TEST = test
        self.ONE_PART = int(NUM * 0.2)
        self.TEST_PART = self.TEST * self.ONE_PART + 1

    def load_train(self):
        train_imgs = []
        train_labels = []
        train_boundary = [[] * SAMPLE_NUM for _ in range(SAMPLE_NUM)]
        train_location=[]
        train_node=[]

        for i in range(1, NUM+1):
            if i == 8 or (i>=self.TEST_PART and i<self.TEST_PART+self.ONE_PART):
                continue
            print('Processing Sample {}'.format(i))

            #读取数据
            brain_dir = self.main_dir + str(i) + self.img_dir
            brain = nib.load(brain_dir).get_data()

            truth = np.zeros(np.shape(brain))
            for j in range(len(self.label_dir)):
                truth_dir = self.main_dir + str(i) + self.label_dir[j]
                truth = truth + (nib.load(truth_dir).get_data()) * (j + 1)
                truth[truth > (j + 1)] = j + 1

            #边界数据读取
            boundary=[None]*SAMPLE_NUM
            for k in range(len(self.boundary_dir)):
                for j in range(SAMPLE_NUM):
                    boundary_dir=self.main_dir + str(i) + self.boundary_dir[k]+str(j)+'.nii.gz'

                    if k==0:
                        boundary[j]=nib.load(boundary_dir).get_data()
                    else:
                        boundary[j] = boundary[j]+nib.load(boundary_dir).get_data()

                    temp=boundary[j]
                    temp[temp>1]=1
                    boundary[j]=temp

            #边界节点读取
            node_dir=self.main_dir + str(i) + self.node_dir
            node=np.load(node_dir)
            node=node-1
            node=np.concatenate((node[:,:1,:]- IMAGE_HIGHT_MIN,node[:,1:,:]- IMAGE_WIDTH_MIN),axis=1)
            node[node < 0] = 0   #裁剪之后，节点坐标改变

            brain = brain[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            truth = truth[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            # 使用全部的slice
            #brain = histogram_equalization(brain)  #数据预处理
            depth = np.shape(brain)[-1]
            for d in range(depth):
                if(d+(params['Batch_size']-1)*params['T2']<depth):
                    brain_one=[]
                    label_one=[]
                    location_one=[]
                    node_one=[]
                    for k in range(params['Batch_size']):
                        brain_one.append(brain[:, :, d + k * params['T2']])
                        label_one.append(truth[:, :, d + k * params['T2']])
                        location_one.append(np.asarray([(d + k * params['T2'])/ depth]))
                        node_one.append(node[:, :, d])

                    train_imgs.append(brain_one)
                    train_labels.append(label_one)

                    # plt.imshow(brain[:, :, d]), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)

                    # c = ['black', 'red']
                    # cmap = mpl.colors.ListedColormap(c)
                    #
                    # plt.imshow(truth[:,:, d],cmap), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)

                    train_location.append(location_one)
                    train_node.append(node_one)

                    # temp=np.zeros((IMAGE_HIGHT,IMAGE_WIDTH))
                    # for point_num in range(params['N']):
                    #     temp[node[point_num,0,d],node[point_num,1,d]]=point_num*1000+10000
                    # # index=tuple(node[:,:,d].transpose((1,0)))
                    # # temp[index]=1
                    #
                    # plt.imshow(temp,cmap=cmap), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)
                    # temp=temp.astype(int)
                    #np.savetxt(str(params['N'])+"_"+str(d)+"t.txt",node[:,:,d])

                    for j in range(SAMPLE_NUM):
                        boundary_one=[]
                        for k in range(params['Batch_size']):
                            boundary_one.append(boundary[j][:,:,d + k * params['T2']])

                        train_boundary[j].append(boundary_one)
                        # plt.imshow(boundary[j][:,:,d]), plt.xticks([]), plt.yticks([])
                        # plt.draw(), plt.show(block=False), plt.pause(0.01)

        #将列表转换成数组
        train_imgs = np.asarray(train_imgs, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.int64)

        for j in range(SAMPLE_NUM):
            train_boundary[j]= np.asarray(train_boundary[j], dtype=np.float32)
        train_location = np.asarray(train_location, dtype=np.float32)
        train_node=np.asarray(train_node,dtype=np.float32)

        # 增加一个维度
        self.train_imgs = np.expand_dims(train_imgs, axis=-1)
        self.train_labels = np.expand_dims(train_labels, axis=-1)
        print(np.shape(self.train_imgs))
        print(np.shape(self.train_labels))

        self.train_boundary=train_boundary
        for j in range(SAMPLE_NUM):
            self.train_boundary[j]= np.expand_dims(self.train_boundary[j], axis=-1)
            print(np.shape(self.train_boundary[j]))

        self.train_location = train_location
        self.train_node=train_node
        print(np.shape(self.train_location))
        print(np.shape(self.train_node))

    def load_valid(self):
        test_imgs = []
        test_labels = []
        test_boundary = [[] * SAMPLE_NUM for _ in range(SAMPLE_NUM)]
        test_location=[]  #存放当前数据属于那一层d*1000/depth
        test_node=[]

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

            # 边界数据读取
            boundary = [None] * SAMPLE_NUM
            for k in range(len(self.boundary_dir)):
                for j in range(SAMPLE_NUM):
                    boundary_dir = self.main_dir + str(i) + self.boundary_dir[k] + str(j) + '.nii.gz'

                    if k == 0:
                        boundary[j] = nib.load(boundary_dir).get_data()
                    else:
                        boundary[j] = boundary[j] + nib.load(boundary_dir).get_data()

                    temp = boundary[j]
                    temp[temp > 1] = 1

            #边界节点读取
            node_dir=self.main_dir + str(i) + self.node_dir
            node=np.load(node_dir)
            node = node - 1
            node = np.concatenate((node[:, :1, :] - IMAGE_HIGHT_MIN, node[:, 1:, :] - IMAGE_WIDTH_MIN), axis=1)
            node[node < 0] = 0  # 裁剪之后，节点坐标改变

            brain = brain[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            truth = truth[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX, :]
            # 使用全部的slice
            #brain = histogram_equalization(brain)  #数据预处理
            depth = np.shape(brain)[-1]
            for d in range(depth):
                if (d + (params['Batch_size'] - 1) * params['T2'] < depth):
                    brain_one = []
                    label_one = []
                    location_one = []
                    node_one = []
                    for k in range(params['Batch_size']):
                        brain_one.append(brain[:, :, d + k * params['T2']])
                        label_one.append(truth[:, :, d + k * params['T2']])
                        location_one.append(np.asarray([(d + k * params['T2'])/ depth]))
                        node_one.append(node[:, :, d])

                    test_imgs.append(brain_one)
                    test_labels.append(label_one)

                    # plt.imshow(brain[:, :, d]), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)
                    #
                    # plt.imshow(truth[:,:, d]), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)

                    test_location.append(location_one)
                    test_node.append(node_one)

                    # temp=np.zeros((IMAGE_HIGHT,IMAGE_WIDTH))
                    # index=tuple(node[:,:,d].transpose((1,0)))
                    # temp[index]=1
                    # plt.imshow(temp), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)

                    for j in range(SAMPLE_NUM):
                        boundary_one = []
                        for k in range(params['Batch_size']):
                            boundary_one.append(boundary[j][:, :, d + k * params['T2']])

                        test_boundary[j].append(boundary_one)
                        # plt.imshow(boundary[j][:,:,d]), plt.xticks([]), plt.yticks([])
                        # plt.draw(), plt.show(block=False), plt.pause(0.01)

        test_imgs = np.asarray(test_imgs, dtype=np.float32)
        test_labels = np.asarray(test_labels, dtype=np.int64)

        for j in range(SAMPLE_NUM):
            test_boundary[j]= np.asarray(test_boundary[j], dtype=np.float32)
        test_location = np.asarray(test_location, dtype=np.float32)
        test_node=np.asarray(test_node,dtype=np.float32)

        # 增加一个维度
        self.test_imgs = np.expand_dims(test_imgs, axis=-1)
        self.test_labels = np.expand_dims(test_labels, axis=-1)
        print(np.shape(self.test_imgs))
        print(np.shape(self.test_labels))

        self.test_boundary=test_boundary
        for j in range(SAMPLE_NUM):
            self.test_boundary[j]= np.expand_dims(self.test_boundary[j], axis=-1)
            print(np.shape(self.test_boundary[j]))

        self.test_location = test_location
        self.test_node=test_node
        print(np.shape(self.test_location))
        print(np.shape(self.test_node))

    def load_test(self):
        test_imgs = []
        test_labels = []
        test_boundary = [[] * SAMPLE_NUM for _ in range(SAMPLE_NUM)]
        test_location = []
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

            # 边界数据读取
            for k in range(len(self.boundary_dir)):
                for j in range(SAMPLE_NUM):
                    boundary_dir = self.main_dir + str(i) + self.boundary_dir[k] + str(j) + '.nii.gz'

                    if k == 0:
                        test_boundary[j] = nib.load(boundary_dir).get_data()
                    else:
                        test_boundary[j] = test_boundary[j] + nib.load(boundary_dir).get_data()

                    temp = test_boundary[j]
                    temp[temp > 1] = 1

            brain=brain[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX,:]
            truth=truth[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX, IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX,:]
            depth=np.shape(brain)[-1]
            ds.append(np.shape(brain)[-1])
            ids.append(i)
            #brain = histogram_equalization(brain)  # 数据预处理
            brain = np.expand_dims(brain, -1)
            truth = np.expand_dims(truth, -1)
            test_imgs.append(brain)
            test_labels.append(truth)
            test_location.append(np.expand_dims(np.arange(depth)/depth, -1))
        return test_imgs, test_labels, ds,ids,test_boundary,test_location

    def get_train_batch(self, batch_size):
        length = len(self.train_imgs)
        while True:
            order = np.random.permutation(length)
            for i in range(length // batch_size):
                batch_x = self.train_imgs[order[i * batch_size]]
                batch_y = self.train_labels[order[i * batch_size]]
                batch_b=[]
                for j in range(SAMPLE_NUM):
                    batch_b.append(self.train_boundary[j][order[i * batch_size]])

                batch_location=self.train_location[order[i * batch_size]]
                batch_node=self.train_node[order[i * batch_size]]
                yield batch_x, batch_y,batch_b,batch_location,batch_node

    # 测试时，使用这个函数，不会打乱顺序
    def get_test_batch(self, batch_size):
        length = len(self.test_imgs)
        while True:
            for i in range(length // batch_size):
                batch_x = self.test_imgs[i * batch_size]
                batch_y = self.test_labels[i * batch_size]

                batch_b = []
                for j in range(SAMPLE_NUM):
                    batch_b.append(self.test_boundary[j][i * batch_size])

                batch_location = self.test_location[i * batch_size]
                batch_node = self.test_node[i * batch_size]
                yield batch_x, batch_y,batch_b,batch_location,batch_node

if __name__ == '__main__':
    data_getter = DataGetterPso(0)
    data_getter.load_train()
    #data_getter.load_valid()
    #data_getter.load_test()

