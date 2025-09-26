# import os, cv2
import SimpleITK as sitk
# from skimage import measure
import numpy as np
import torch
import torch.nn.functional as F

from ..tools.preprocess_scan_and_label import set_window, bone_thres, connect_check, fill3d
# 先获得bone_gray_image, del_bone_mask

def del_surplus(bone_mask, image):
    """
    提取肺实质的边界框
    :param mask: mask
    :param image:  影像
    :return mask_del: 删减无关区域后的mask
    :return image_del: 删减无关无区域骺的image
    :return 删减的边界框
    """
    zs, ys, xs = np.where(bone_mask)
    zmin, zmax = np.min(zs), np.max(zs)
    ymin, ymax = np.min(ys), np.max(ys)
    xmin, xmax = np.min(xs), np.max(xs)

    bone_del = bone_mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    # rib_del = rib_seg[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    gray_image_del = image[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]

    return bone_del,  gray_image_del.astype(np.int32), (zmin, ymin, xmin, zmax, ymax, xmax)


def get_rib_preprocess(hu_img):
    # 先设置窗宽窗位、然后骨头阈值分割、连通区域、肺实质删除
    gray_img = set_window(hu_image=hu_img, min_v=-400, max_v=1000)
    bone_binary = bone_thres(gray_img=gray_img, gray_threshold=90)
    bone_mask = connect_check(binary_img=bone_binary)
    del_bone_msk, del_gray_img, del_boundbox = del_surplus(bone_mask=bone_mask, image=gray_img)
    # del_boundbox是删减的边界框
    return del_bone_msk, del_gray_img, del_boundbox


def crop_cube(gray_img_arr, cube_size):
    """
    滑窗式裁剪cube
    :param gray_img_arr: 待裁剪的影像数组
    :param cube_size: 参数如（64， 64， 64）, 裁剪的cube的尺寸
    :return: cube, (zmin, ymin, xmin, zmax, ymax, xmax)
    """
    depth, height, width = gray_img_arr.shape
    cube_z, cube_y, cube_x = cube_size
    for z in range(0, depth, int(cube_z / 2)):  # 以一半的z尺寸为步长
        for y in range(0, height, int(cube_y / 2)):
            for x in range(0, width, int(cube_x / 2)):
                cube = np.zeros(cube_size, dtype=np.float32)  # 根据cube_size临时构建的全零数组
                # 前者每次循环的起始位置z， 后者得到了结束位置，如果结束位置超出bone_gray_image的大小，就截取到3D图像的最大z值
                zmin, zmax = z, min(z + cube_z, depth)
                ymin, ymax = y, min(y + cube_y, height)
                xmin, xmax = x, min(x + cube_x, width)
                # 依次遍历每个块，将bone_gray_image对应坐标中的数据存储到tmp_img_cube
                cube[:(zmax - zmin), :(ymax - ymin), :(xmax - xmin)] = gray_img_arr[zmin:zmax,
                                                                               ymin:ymax,
                                                                               xmin:xmax]
                cube_arr = np.expand_dims(cube, 0)
                cube_arr = np.expand_dims(cube_arr, 0)

                yield cube_arr, (zmin, ymin, xmin, zmax, ymax, xmax)


