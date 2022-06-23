import numpy as np
import os
import nibabel as nib
import cv2
from skimage import transform
import random
import matplotlib.pyplot as plt

#边界上均匀采样n个点
def SelectPoint(image, T, N):
    deter=0
    label=np.zeros((image.shape[0],N,2),dtype=np.int16)
    for i in range(image.shape[0]):
        if 1 in image[i,:, :]:
            #print(i)
            deter = 1
            # 1.根据真实标签得到边界节点，逆时针
            label_slice = image[i,:, :].copy()
            label_slice = label_slice.astype(np.uint8)
            #plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
            #plt.draw(), plt.show(block=False), plt.pause(0.01)

            _,contours, hierarchy = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for k in range(1,len(contours)):
                contours[0]=np.concatenate((contours[0],contours[k]),axis=0)

            contours=contours[0]
            contours=np.reshape(contours,[contours.shape[0],2])

            #可选取点数<N,采用插值的方式
            while (contours.shape[0]<N):
                contours=np.repeat(contours,2,axis=0)
                contours=(contours+np.concatenate((contours[1:,:],contours[:1,:]),axis=0))//2

            contours=np.concatenate((contours[:,1:],contours[:,:1]),axis=-1)
            mid_x=(np.min(contours[:,0])+np.max(contours[:,0]))//2

            # 2.根据边界节点，确定起始元素，中轴线的上面元素
            index=np.argwhere(contours[:,0]==mid_x)
            while(index.size==0):
                mid_x=mid_x+1
                index = np.argwhere(contours[:, 0] == mid_x)

            mid_y=-1
            index_start=-1
            for j in range(index.size):
                if (mid_y==-1 or contours[index[j][0],1]<mid_y):
                    mid_y=contours[index[j][0],1]
                    index_start=index[j][0]

            contours=np.concatenate((contours[index_start:,:],contours[:index_start,:]),axis=0)

            # 3.迭代在边界上随机选取点,选取均匀采样的点
            DIS_MIN=-1
            for j in range(T):
                randindex=np.array(random.sample(range(1,contours.shape[0]),N-1))
                randindex=np.concatenate(([0],np.sort(randindex)))

                point_0=contours[randindex,:]
                point_1=np.concatenate((point_0[1:,:],point_0[:1,:]),axis=0)

                # 4.计算相邻节点的距离，求最大距离和最小距离之差
                dis=(point_0-point_1)*(point_0-point_1)
                dis=dis[:,0]+dis[:,1]
                dis=np.max(dis)-np.min(dis)

                if(DIS_MIN==-1 or dis<DIS_MIN):
                    DIS_MIN=dis
                    label[i,:, :]=0
                    label[i,:, :]=point_0

    return label,deter

def SelectPoint1(image, T, N):
    label=np.zeros((N,2,image.shape[2]),dtype=np.int16)
    for i in range(image.shape[-1]):
        if 1 in image[:, :, i]:
            #print(i)
            # 1.根据真实标签得到边界节点，逆时针
            label_slice = image[:, :, i].copy()
            label_slice = label_slice.astype(np.uint8)
            #plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
            #plt.draw(), plt.show(block=False), plt.pause(0.01)

            contours, hierarchy = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for k in range(1,len(contours)):
                contours[0]=np.concatenate((contours[0],contours[k]),axis=0)

            contours=contours[0]
            contours=np.reshape(contours,[contours.shape[0],2])

            #可选取点数<N,采用插值的方式
            while (contours.shape[0]<N):
                contours=np.repeat(contours,2,axis=0)
                contours=(contours+np.concatenate((contours[1:,:],contours[:1,:]),axis=0))//2

            contours=np.concatenate((contours[:,1:],contours[:,:1]),axis=-1)
            mid_x=(np.min(contours[:,0])+np.max(contours[:,0]))//2

            # 2.根据边界节点，确定起始元素，中轴线的上面元素
            index=np.argwhere(contours[:,0]==mid_x)
            while(index.size==0):
                mid_x=mid_x+1
                index = np.argwhere(contours[:, 0] == mid_x)

            mid_y=-1
            index_start=-1
            for j in range(index.size):
                if (mid_y==-1 or contours[index[j][0],1]<mid_y):
                    mid_y=contours[index[j][0],1]
                    index_start=index[j][0]

            contours=np.concatenate((contours[index_start:,:],contours[:index_start,:]),axis=0)

            # 3.迭代在边界上随机选取点,选取均匀采样的点
            DIS_MIN=-1
            for j in range(T):
                randindex=np.array(random.sample(range(1,contours.shape[0]),N-1))
                randindex=np.concatenate(([0],np.sort(randindex)))

                point_0=contours[randindex,:]
                point_1=np.concatenate((point_0[1:,:],point_0[:1,:]),axis=0)

                # 4.计算相邻节点的距离，求最大距离和最小距离之差
                dis=(point_0-point_1)*(point_0-point_1)
                dis=dis[:,0]+dis[:,1]
                dis=np.max(dis)-np.min(dis)

                if(DIS_MIN==-1 or dis<DIS_MIN):
                    DIS_MIN=dis
                    label[:,:,i]=0
                    label[:,:,i]=point_0

    return label

if __name__=='__main__':
    T = 40000
    N = 256

    DATA_PATH = "/data/targetarea/dataset/"
    IMAGE_DATA_PATH = "/label_ctv_crop.nii.gz"
    SAVE_PATH = "/label_point_crop_"+str(T)+"_"+str(N)

    for i in range(60,70):
        print("process "+str(i))
        img_path = DATA_PATH + str(i) + IMAGE_DATA_PATH
        save_path = DATA_PATH + str(i) + SAVE_PATH
        if os.path.isfile(img_path):
            img = nib.load(img_path).get_data()
            #label=np.load("F:/data/targetarea/dataset/1/label_point_crop_40000_128.npy")
            np.save(save_path+".npy",SelectPoint1(img,T,N))