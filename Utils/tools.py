import os
import nibabel as nib
import cv2
import numpy as np
from skimage import transform
import math
import matplotlib.pyplot as plt

def bounding_box():
    # 查看组织的大致范围
    DATA_PATH = "F:/data/targetarea/dataset/"
    IMAGE_DATA_PATH = "/label_ctv_crop.nii.gz"

    IMAGE_HIGHT_MIN = 512
    IMAGE_WIDTH_MIN = 512
    IMAGE_DEPTH_MIN = 512
    IMAGE_HIGHT_MAX = 0
    IMAGE_WIDTH_MAX = 0
    IMAGE_DEPTH_MAX = 0

    for i in range(1, 66):
        img_path = DATA_PATH + str(i) + IMAGE_DATA_PATH
        if os.path.isfile(img_path):
            img = nib.load(img_path).get_data()

            non_zero = np.nonzero(img)

            HIGHT_MIN = np.min(non_zero[0])
            WIDTH_MIN = np.min(non_zero[1])
            HIGHT_MAX = np.max(non_zero[0])
            WIDTH_MAX = np.max(non_zero[1])

            if HIGHT_MIN < IMAGE_HIGHT_MIN:
                IMAGE_HIGHT_MIN = HIGHT_MIN

            if WIDTH_MIN < IMAGE_WIDTH_MIN:
                IMAGE_WIDTH_MIN = WIDTH_MIN

            if HIGHT_MAX > IMAGE_HIGHT_MAX:
                IMAGE_HIGHT_MAX = HIGHT_MAX

            if WIDTH_MAX > IMAGE_WIDTH_MAX:
                IMAGE_WIDTH_MAX = WIDTH_MAX

            print(str(i) + ": " + str(HIGHT_MIN) + "   " + str(WIDTH_MIN) + "   " + str(HIGHT_MAX) + "    " + str(
                WIDTH_MAX))
    print(str(IMAGE_HIGHT_MIN) + "   " + str(IMAGE_WIDTH_MIN) + "   " + str(IMAGE_HIGHT_MAX) + "    " + str(
        IMAGE_WIDTH_MAX))

#寻找图像的轮廓,rate标签压缩几倍后的轮廓
def boundary(rate=5):
    DATA_PATH = "/data/targetarea/dataset/"
    IMAGE_DATA_PATH = "/label_bone_crop.nii.gz"
    SAVE_PATH = "/label_bone_boundary_"

    IMAGE_HIGHT_MIN = 90
    IMAGE_WIDTH_MIN = 4
    IMAGE_HIGHT_MAX = 298
    IMAGE_WIDTH_MAX = 196

    # 找出靶区的边界
    for i in range(1, 66):
        img_path = DATA_PATH + str(i) + IMAGE_DATA_PATH
        save_path = DATA_PATH + str(i) + SAVE_PATH
        print(i)
        if os.path.isfile(img_path):
            img = nib.load(img_path).get_data()
            img_affine = nib.load(img_path).affine
            img=img[IMAGE_HIGHT_MIN:IMAGE_HIGHT_MAX,IMAGE_WIDTH_MIN:IMAGE_WIDTH_MAX,:]
            depth = img.shape[-1]

            for j in range(depth):
                if 1 in img[:, :, j]:
                    label_slice = img[:, :, j]
                    label_slice = label_slice.astype(np.uint8)
                    contours, hierarchy = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    temp = np.zeros((label_slice.shape[0], label_slice.shape[1]), np.uint8)
                    for k in range(len(contours)):
                        temp = cv2.drawContours(temp, contours, k, 1, 3)

                    img[:, :, j] = temp

            #图像进行下采样
            w=img.shape[0]
            h=img.shape[1]
            for r in range(rate):
                data = transform.resize(
                    image=img,
                    output_shape=(w,h,depth)
                )

                data_min=np.min(data)
                data[data> data_min]=1
                data[data<=data_min]=0

                label_nii = nib.Nifti1Image(data, img_affine)
                nib.save(label_nii, save_path+str(r)+".nii.gz")

                w=w//2
                h=h//2

def merge_boundary(rate=5):
    DATA_PATH = 'F:/data/targetarea/dataset/'
    BOUNDARY_PATH=''
    SAVE_PATH = "label_merge_boundary_"

    for i in range(1,66):
        gt_label_path = GT_DATA_PATH + str(i) + GT_LABEL_DATA_PATH
        pred_label_path=PRED_DATA_PATH+str(i)+PRED_LABEL_DATA_PATH
        save_path = PRED_DATA_PATH + str(i)+SAVE_PATH
        if os.path.isfile(pred_label_path):
            img = nib.load(gt_label_path).get_data()
            img_affine = nib.load(gt_label_path).affine
            img2 = nib.load(pred_label_path).get_data()
            depth = img.shape[-1]

            for j in range(depth):
                if 1 in img[:, :, j]:
                    label_slice = img[:, :, j]
                    # plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)
                    label_slice = label_slice.astype(np.uint8)
                    contours, hierarchy = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    pre_label_slice = img2[:, :, j]
                    pre_label_slice = pre_label_slice.astype(np.uint8)
                    contours2, hierarchy2 = cv2.findContours(pre_label_slice, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                    temp = np.zeros((label_slice.shape[0], label_slice.shape[1]), np.uint8)
                    for k in range(len(contours)):
                        temp = cv2.drawContours(temp, contours, k, 1, 2)

                    for k in range(len(contours2)):
                        temp = cv2.drawContours(temp, contours2, k, 2, 2)

                    img[:, :, j] = temp

            label_nii = nib.Nifti1Image(img, img_affine)
            nib.save(label_nii, save_path)

def mat_math(intput, str,img):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output

# CV函数
def CV(LSF, img, mu, nu, epison, step):
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan",img))
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix * Ix + Iy * Iy, "sqrt",img)
    Nx = Ix / (s + 0.000001)
    Ny = Iy / (s + 0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy
    Length = nu * Drc * cur

    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu * (Lap - cur)

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

    LSF = LSF + step * (Length + Penalty + CVterm)
    # plt.imshow(s, cmap ='gray'),plt.show()
    return LSF

#CT图像的背景
def background():
    DATA_PATH = "/data/targetarea/dataset/"
    IMAGE_DATA_PATH = "/ct.nii.gz"
    SAVE_PATH = "/label_background.nii.gz"

    # 将分割部分的脂肪分割出来
    for i in range(64, 65):
        img_path = DATA_PATH + str(i) + IMAGE_DATA_PATH
        save_path = DATA_PATH + str(i) + SAVE_PATH
        if os.path.isfile(img_path):
            img = nib.load(img_path).get_data()
            img_affine = nib.load(img_path).affine
            depth = img.shape[-1]

            # 第一步，分割出外部区域
            for j in range(depth):
                # print(j)
                label_slice = img[:, :, j]
                label_slice[label_slice < -300] = np.min(label_slice)
                #plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
                #plt.draw(), plt.show(block=False), plt.pause(0.01)

                # 初始水平集函数
                IniLSF = np.ones((label_slice.shape[0], label_slice.shape[1]), label_slice.dtype)
                IniLSF[260:320, 260:320] = -1
                IniLSF = -IniLSF

                # 模型参数
                mu = 1
                nu = 0.003 * 512 * 512
                num = 5
                epison = 1
                step = 0.1
                LSF = IniLSF
                for k in range(1, num):
                    LSF = CV(LSF, label_slice, mu, nu, epison, step)  # 迭代
                    #if k % 1 == 0:  # 显示分割轮廓
                        #plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
                        #plt.contour(LSF, [0], colors='r', linewidth=2)
                        #plt.draw(), plt.show(block=False), plt.pause(0.01)

                LSF[LSF >= 0] = 1
                LSF[LSF < 0] = 0

                #if (j >= 114):
                #    LSF = 1 - LSF

                #plt.imshow(LSF), plt.xticks([]), plt.yticks([])
                #plt.draw(), plt.show(block=False), plt.pause(0.01)
                # 开闭运算
                kernel = np.ones((20, 20), np.uint8)
                LSF = cv2.erode(LSF, kernel)
                LSF = cv2.dilate(LSF, kernel)
                LSF = np.array(LSF, np.uint8)
                LSF = cv2.fastNlMeansDenoising(LSF, None, 1, 7, 21)
                kernel = np.ones((5, 5), np.uint8)
                LSF = cv2.erode(LSF, kernel)
                #plt.imshow(LSF), plt.xticks([]), plt.yticks([])
                #plt.draw(), plt.show(block=False), plt.pause(0.01)

                LSF = LSF.astype(np.uint8)
                contours, hierarchy = cv2.findContours(LSF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 寻找面积最大的区域的轮廓
                area = []
                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))

                max_idx = np.argmax(area)
                LSF[:, :] = 1
                LSF = cv2.drawContours(LSF, contours, max_idx, 0, -1)

                #plt.imshow(LSF), plt.xticks([]), plt.yticks([])
                #plt.draw(), plt.show(block=False), plt.pause(0.01)
                LSF[LSF == 1] = 10
                img[:, :, j] = LSF

            label_nii = nib.Nifti1Image(img, img_affine)
            nib.save(label_nii, save_path)

def crop_size():
    DATA_PATH = "/data/targetarea/dataset/"
    IMAGE_DATA_PATH = "/label_background.nii.gz"

    IMAGE_HIGHT = 0
    IMAGE_WIDTH = 0

    # 找出靶区的边界
    for i in range(1, 66):
        img_path = DATA_PATH + str(i) + IMAGE_DATA_PATH
        if os.path.isfile(img_path):
            img = nib.load(img_path).get_data()

            img[img!=0]=1
            img=1-img
            non_zero=np.nonzero(img)
            image_hight=np.max(non_zero[0])-np.min(non_zero[0])
            image_width = np.max(non_zero[1]) - np.min(non_zero[1])

            if image_hight>IMAGE_HIGHT:
                IMAGE_HIGHT=image_hight

            if image_width>IMAGE_WIDTH:
                IMAGE_WIDTH=image_width

            print(str(i) + ": " + str(image_hight) + "   " + str(image_width))
    print(str(IMAGE_HIGHT) + "   " + str(IMAGE_WIDTH))

#将CT图像进行裁剪，保留具有图像部分
def crop():
    DATA_PATH = "/data/targetarea/dataset/"
    IMAGE_DATA_PATH = "/label_bone.nii.gz"
    LABEL_DATA_PATH="/label_background.nii.gz"
    SAVE_PATH = "/label_bone_crop.nii.gz"

    # 找出靶区的边界
    for i in range(20, 21):
        #img_path = DATA_PATH + str(i) + IMAGE_DATA_PATH
        label_path = DATA_PATH + str(i) + LABEL_DATA_PATH
        #save_path = DATA_PATH + str(i) + SAVE_PATH
        img_path = "F:/data/targetarea/result-contrast/dd_resnet/1/20_pre_label_ctv.nii.gz"
        save_path = "F:/data/targetarea/result-contrast/dd_resnet/1/20_pre_label_ctv_crop.nii.gz"
        if os.path.isfile(img_path):
            print("process:"+str(i))
            img = nib.load(img_path).get_data()
            label=nib.load(label_path).get_data()
            img_affine = nib.load(img_path).affine
            depth = img.shape[-1]

            label[label!=0]=1
            label=1-label
            non_zero=np.nonzero(label)
            cen_x = (np.max(non_zero[0]) + np.min(non_zero[0]))//2
            cen_y = (np.max(non_zero[1]) + np.min(non_zero[1]))//2

            #img=img[cen_x-IMAGE_HIGHT//2:cen_x+IMAGE_HIGHT//2,cen_y-IMAGE_WIDTH//2:cen_y+IMAGE_WIDTH//2,:]
            img = img[np.min(non_zero[0]):np.max(non_zero[0])+1,np.min(non_zero[1]):np.max(non_zero[1])+1, :]

            label_nii = nib.Nifti1Image(img[90:298,4:196,:], img_affine)
            nib.save(label_nii, save_path)

#将真实值和预测值进行合并显示
def merge_label():
    GT_DATA_PATH = 'F:/data/targetarea/result/8/1/'
    PRED_DATA_PATH='F:/data/targetarea/result/8/1/'
    GT_LABEL_DATA_PATH = "_label_ctv.nii.gz"
    PRED_LABEL_DATA_PATH="_pre_label_ctv.nii.gz"
    SAVE_PATH = "_merge_label_ctv.nii.gz"

    for i in range(20,21):
        gt_label_path = GT_DATA_PATH + str(i) + GT_LABEL_DATA_PATH
        pred_label_path=PRED_DATA_PATH+str(i)+PRED_LABEL_DATA_PATH
        save_path = PRED_DATA_PATH + str(i)+SAVE_PATH
        if os.path.isfile(pred_label_path):
            img = nib.load(gt_label_path).get_data()
            img_affine = nib.load(gt_label_path).affine
            img2 = nib.load(pred_label_path).get_data()
            depth = img.shape[-1]

            for j in range(depth):
                if 1 in img[:, :, j]:
                    label_slice = img[:, :, j]
                    # plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
                    # plt.draw(), plt.show(block=False), plt.pause(0.01)
                    label_slice = label_slice.astype(np.uint8)
                    contours, hierarchy = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    pre_label_slice = img2[:, :, j]
                    pre_label_slice = pre_label_slice.astype(np.uint8)
                    contours2, hierarchy2 = cv2.findContours(pre_label_slice, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                    temp = np.zeros((label_slice.shape[0], label_slice.shape[1]), np.uint8)
                    for k in range(len(contours)):
                        temp = cv2.drawContours(temp, contours, k, 1, 2)

                    for k in range(len(contours2)):
                        temp = cv2.drawContours(temp, contours2, k, 2, 2)

                    img[:, :, j] = temp

            label_nii = nib.Nifti1Image(img, img_affine)
            nib.save(label_nii, save_path)

#直接将多个label混合起来
def mergeboundary():
    DATA_PATH = 'F:/data/targetarea/dataset/'
    BONE_DATA_PATH="/label_bone_boundary_0.nii.gz"
    PSOAS_DATA_PATH = "/label_psoas_major_boundary_0.nii.gz"
    CTV_DATA_PATH="/label_ctv_boundary_0.nii.gz"
    SAVE_PATH = "/label_merge_boundary.nii.gz"

    for i in range(1,2):
        bone_label_path = DATA_PATH +str(i)+ BONE_DATA_PATH
        psoas_label_path=DATA_PATH +str(i)+ PSOAS_DATA_PATH
        ctv_label_path = DATA_PATH +str(i)+ CTV_DATA_PATH
        save_path = DATA_PATH+str(i)+SAVE_PATH
        if os.path.isfile(bone_label_path):
            img = nib.load(bone_label_path).get_data()
            img_affine = nib.load(bone_label_path).affine
            psoas = (nib.load(psoas_label_path).get_data())*2
            ctv = (nib.load(ctv_label_path).get_data())*3
            depth = img.shape[-1]

            img=img+psoas
            img[img>2]=2

            img=img+ctv
            img[img>3]=3

            label_nii = nib.Nifti1Image(img, img_affine)
            nib.save(label_nii, save_path)

if __name__=='__main__':
    #background()
    #crop_size()
    #crop()
    #bounding_box()
    #boundary()
    merge_label()
    #mergeboundary()

    GT_DATA_PATH = 'F:/data/targetarea/result/8/1/20_label_ctv.nii.gz'
    PRED_DATA_PATH = 'F:/data/targetarea/result-contrast/segnet/1/20_pre_label_ctv_crop.nii.gz'
    SAVE_PATH = 'F:/data/targetarea/result-contrast/segnet/1/20_pre_merge_label_ctv_crop.nii.gz'
    BOUNDARY_SAVE_PATH='F:/data/targetarea/result-contrast/segnet/1/20_pre_merge_boundary_label_ctv_crop.nii.gz'

    gt_label_path = GT_DATA_PATH
    pred_label_path = PRED_DATA_PATH
    save_path = SAVE_PATH
    # if os.path.isfile(pred_label_path):
    #     img = nib.load(gt_label_path).get_data()
    #     img_affine = nib.load(gt_label_path).affine
    #     img2 = nib.load(pred_label_path).get_data()
    #     depth = img.shape[-1]
    #
    #     img=img[:,:,:,0]
    #
    #     img=img+img2*2
    #
    #     label_nii = nib.Nifti1Image(img, img_affine)
    #     nib.save(label_nii, save_path)

    if os.path.isfile(pred_label_path):
        img = nib.load(gt_label_path).get_data()
        img_affine = nib.load(gt_label_path).affine
        img2 = nib.load(pred_label_path).get_data()

        img = img[:, :, :, 0]
        depth = img.shape[-1]

        for j in range(depth):
            if 1 in img[:, :, j]:
                label_slice = img[:, :, j]
                #plt.imshow(label_slice), plt.xticks([]), plt.yticks([])
                #plt.draw(), plt.show(block=False), plt.pause(0.01)
                label_slice = label_slice.astype(np.uint8)
                contours, hierarchy = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                pre_label_slice = img2[:, :, j]
                pre_label_slice = pre_label_slice.astype(np.uint8)
                contours2, hierarchy2 = cv2.findContours(pre_label_slice, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                temp = np.zeros((label_slice.shape[0], label_slice.shape[1]), np.uint8)
                for k in range(len(contours)):
                    temp = cv2.drawContours(temp, contours, k, 1, 2)

                for k in range(len(contours2)):
                    temp = cv2.drawContours(temp, contours2, k, 2, 2)

                img[:, :, j] = temp

        label_nii = nib.Nifti1Image(img, img_affine)
        nib.save(label_nii, BOUNDARY_SAVE_PATH)