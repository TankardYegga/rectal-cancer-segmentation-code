import numpy as np
from skimage import measure

#边界上均匀采样n个点
def postprocess(image, cate):

    top=0
    down=0

    for i in range(image.shape[-1]):
        if(cate[0,0,i]>=0.5):
            top=i
            break

    for i in range(image.shape[-1]-1,-1,-1):
        if(cate[0,0,i]>=0.5):
            down=i
            break

    label, num = measure.label(image, connectivity=1, return_num=True)

    region = measure.regionprops(label)
    area_list = [region[k].area for k in range(num)]
    index_max = area_list.index(max(area_list)) + 1
    label[label != index_max] = 0
    label[label == index_max] = 1
    label[:,:,:top]=0
    label[:, :, down+1:] = 0
    label = label.astype(np.uint8)

    return label
