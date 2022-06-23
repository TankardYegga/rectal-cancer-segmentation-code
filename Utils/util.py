"""
-*- coding: utf-8 -*-
 @Time      : 18-3-21 下午2:21
 @Author    : Philex Wu
 @File      : util.py
 @Project   : Network
 @Software  : PyCharm
 工具模块
"""

import numpy as np
np.set_printoptions(suppress=True)
import copy
import nibabel
import SimpleITK as sitk

# 图片填充
def imgs_fill(imgs, test_samples=13):
    new_imgs = np.zeros([128 * test_samples, 256, 256, 4])
    new_imgs[:, 32:224, 32:224, :] = imgs
    return new_imgs

# 将内容写入文件
def write_to_file(content, filename, dir='/home/philex/Desktop/workshop/yyt_segnet/Records/'):
    file_dir = dir + filename
    with open(file_dir, 'a+') as f:
        f.writelines(content)
        f.write('\n')

# 保留指定类别的label,用于显示
def show_selected_classes(label_dir, image_dir, save_dir, classes):
    data = nibabel.load(label_dir).get_data()
    affine = nibabel.load(image_dir).affine
    count = 1
    for cla in classes:
        data[data == count] = 0
        data[data == cla] = count
        count += 1
    data[data >= count] = 0
    print(np.max(data))
    print(np.min(data))
    data_nii = nibabel.Nifti1Image(data, affine)
    nibabel.save(data_nii, save_dir)

#分割指标dsc
def dsc_similarity_coef(pred, label, argmax=True, num_classes=4):
    if argmax:
        pred = np.argmax(pred, axis=-1)
        label = np.argmax(label, axis=-1)
    shape = np.shape(pred)

    pred_o = np.reshape(pred, [shape[0], shape[1] * shape[2]])
    label_o = np.reshape(label, [shape[0], shape[1] * shape[2]])
    dscs = []

    for i in range(1, num_classes):
        # not using copy is only address and change the origin value
        seg = copy.copy(pred_o)
        gt = copy.copy(label_o)
        seg[seg != i] = 0
        gt[gt != i] = 0
        seg[seg == i] = 1
        gt[gt == i] = 1

        insection = sum(np.sum(seg * gt, axis=-1))
        #print('insection', insection)
        sum1 = sum(np.sum(seg, axis=-1) + np.sum(gt, axis=-1))
        dsc_i = 2 * insection / sum1

        #print('dsc_{}    :{:.5f}'.format(i, dsc_i))
        dscs.append(dsc_i)

    return dscs

#分割指标avd
def avd_similarity_coef(pred, label, argmax=True, num_classes=4):
    if argmax:
        pred = np.argmax(pred, axis=-1)
        label = np.argmax(label, axis=-1)
    shape = np.shape(pred)

    pred_o = np.reshape(pred, [shape[0], shape[1] * shape[2]])
    label_o = np.reshape(label, [shape[0], shape[1] * shape[2]])
    avds = []

    for i in range(1, num_classes):
        # not using copy is only address and change the origin value
        seg = copy.copy(pred_o)
        gt = copy.copy(label_o)
        seg[seg != i] = 0
        gt[gt != i] = 0
        seg[seg == i] = 1
        gt[gt == i] = 1

        sum_seg=np.sum(seg)
        sum_gt=np.sum(gt)
        avd_i = abs(sum_seg-sum_gt) / sum_gt

        avds.append(avd_i)

    return avds

#分割指标hd
def hd_similarity_coef(pred, label, argmax=True, num_classes=4):
    if argmax:
        pred = np.argmax(pred, axis=-1)
        label = np.argmax(label, axis=-1)
    shape = np.shape(pred)

    pred_o = np.reshape(pred, [shape[0], shape[1] * shape[2]])
    label_o = np.reshape(label, [shape[0], shape[1] * shape[2]])
    hds = []

    for i in range(1, num_classes):
        # not using copy is only address and change the origin value
        seg = copy.copy(pred_o)
        gt = copy.copy(label_o)
        seg[seg != i] = 0
        gt[gt != i] = 0
        seg[seg == i] = 1
        gt[gt == i] = 1

        pred_image = sitk.GetImageFromArray(pred)
        label_image = sitk.GetImageFromArray(label)
        hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
        hausdorffcomputer.Execute(pred_image, label_image)
        hd_i = hausdorffcomputer.GetHausdorffDistance()

        hds.append(hd_i)
    return hds

#分割指标ppv
def ppv_similarity_coef(pred, label, argmax=True, num_classes=4):
    if argmax:
        pred = np.argmax(pred, axis=-1)
        label = np.argmax(label, axis=-1)
    shape = np.shape(pred)

    pred_o = np.reshape(pred, [shape[0], shape[1] * shape[2]])
    label_o = np.reshape(label, [shape[0], shape[1] * shape[2]])
    ppvs = []

    for i in range(1, num_classes):
        # not using copy is only address and change the origin value
        seg = copy.copy(pred_o)
        gt = copy.copy(label_o)
        seg[seg != i] = 0
        gt[gt != i] = 0
        seg[seg == i] = 1
        gt[gt == i] = 1

        insection = sum(np.sum(seg * gt, axis=-1))
        #print('insection', insection)
        sum1 = np.sum(seg)
        ppv_i = insection / sum1

        #print('dsc_{}    :{:.5f}'.format(i, dsc_i))
        ppvs.append(ppv_i)

    return ppvs