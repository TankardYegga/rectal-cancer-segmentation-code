import os
import numpy as np
import cv2
import nibabel as nib
from skimage import measure

def bone_seg(data_path,data_nii):
    for i in range(31,66):
        one_img_path = data_path + str(i) + data_nii
        img = nib.load(one_img_path).get_data()
        for j in range(img.shape[-1]):
            label_slice=img[:,:,j].copy()
            ret, label_slice = cv2.threshold(label_slice, 150, 3071, cv2.THRESH_BINARY)
            label_slice[label_slice != 0] = 1
            label_slice = label_slice.astype(np.uint8)
            contours, _ = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(label_slice.shape, np.uint8)
            for contour in contours:
                cv2.fillPoly(mask, [contour], 1)
            img[:, :, j]=mask

        label,num=measure.label(img,connectivity=2,return_num=True)

        region=measure.regionprops(label)
        area_list=[region[k].area for k in range(num)]
        index_max=area_list.index(max(area_list))+1
        label[label!=index_max] = 0
        label[label == index_max] = 1
        label = label.astype(np.uint8)

        save_nii = nib.Nifti1Image(label,nib.load(one_img_path).affine)
        nib.save(save_nii, data_path + str(i) +'/label_bone1.nii.gz')

if __name__ == '__main__':
    data_path = 'F:/data/targetarea/dataset/'
    data_nii = '/ct.nii.gz'
    bone_seg(data_path=data_path,data_nii=data_nii)


