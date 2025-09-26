import os.path
import random
import math
import torch
import numpy as np
import shutil
from scipy import ndimage
from collections import defaultdict
from preprocess_rib_seg import crop_cube
from ..tools.preprocess_scan_and_label import set_window, bone_thres, connect_check
import SimpleITK as sitk
from tqdm import tqdm

LABEL_PATH = "/data/yyh/RIB_SEG/ribseg/"
IMG_PATH = "/data/yyh/RIB_SEG/series/"
SAVE_PATH = "/data/yyh/Processed Data/cube_1/"

# cube的尺寸
WEIGHT = 64
HEIGHT = 64
QUEUE = 64
#另一个cube的偏移
RATIO_RANGE = (0.7, 0.9)
SHIFTING = 0.7

def random_two_crop(img, position_arr):
    """
    :param img: 需要裁剪的图像
    :param position: (x,y,z)切割的坐标
    :return: img_stack=torch.stack(img1,img2),img1为根据坐标切割出来的cube，img2为根据img1的位置偏移一定量的cube
    """
    img1_arr = []
    img2_arr = []
    for position in position_arr:
        z1_min, y1_min, x1_min, z1_max, y1_max, x1_max = position

        ratio_l, ratio_h = RATIO_RANGE
        random_ratio = ratio_l + random.random() * (ratio_h - ratio_l)
        w, h, q = math.floor(random_ratio * WEIGHT), math.floor(random_ratio * HEIGHT), math.floor(random_ratio * QUEUE)

        x2_min = x1_min + math.floor((WEIGHT - w) * random.random())
        y2_min = y1_min + math.floor((HEIGHT - h) * random.random())
        z2_min = z1_min + math.floor((QUEUE - q) * random.random())

        # x2_min = x1_min + SHIFTING * WEIGHT
        # y2_min = y1_min + SHIFTING * WEIGHT
        # z2_min = z1_min + SHIFTING * QUEUE

        x2_max = x2_min + WEIGHT
        y2_max = y2_min + HEIGHT
        z2_max = z2_min + QUEUE

        img1 = img[z1_min:z1_max, y1_min:y1_max, x1_min:x1_max]
        # OrthoSlicer3D(img1).show()
        img2 = img[z2_min:z2_max, y2_min:y2_max, x2_min:x2_max]
        # OrthoSlicer3D(img2).show()
        if (img1.shape == (QUEUE, HEIGHT, WEIGHT) and img2.shape == (QUEUE, HEIGHT, WEIGHT)):
            img1_arr.append(img1)
            img2_arr.append(img2)
    return img1_arr, img2_arr

def del_surplus_noMask(bone, image):
    """
    提取肺实质的边界框
    :param bone: 骨分割mask
    :param rib: 肋骨分割标注mask
    :param image:  影像
    :return mask_del: 删减无关区域后的骨分割mask
    :return rib_del: 删减无关区域的后的肋骨分割标注mask
    :return image_del: 删减无关无区域骺的image
    :return 删减的边界框
    """
    zs, ys, xs = np.where(bone)
    zmin, zmax = np.min(zs), np.max(zs)
    ymin, ymax = np.min(ys), np.max(ys)
    xmin, xmax = np.min(xs), np.max(xs)

    bone_del = bone[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    image_del = image[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]

    return bone_del, image_del.astype(np.int32), (zmin, ymin, xmin, zmax, ymax, xmax)


def img_preprocess(path):
    hu_img = torch.from_numpy(read_img(path=path))
    print('hu_img:', hu_img.shape)
    gray_img = set_window(hu_image=hu_img, min_v=-400, max_v=1000)
    bone_binary = bone_thres(gray_img=gray_img, gray_threhold=90)
    bone_binary = connect_check(bone_binary)
    del_bone_msk, del_rib_img, del_bndbox = del_surplus_noMask(bone=bone_binary, image=gray_img)
    del_rib_img *= del_bone_msk

    # OrthoSlicer3D(del_bone_msk).show()
    cube = crop_cube(del_bone_msk, (QUEUE, HEIGHT, WEIGHT))
    position_arr = []
    for (img, position) in cube:
        img = torch.squeeze(torch.Tensor(img))
        value, num = torch.unique(img, return_counts=True)
        if len(num.numpy()) > 1:
            if 15000 < num.numpy()[1]:
                # OrthoSlicer3D(img).show()
                position_arr.append(position)

    img1_arr, img2_arr = random_two_crop(del_rib_img, position_arr)
    print('img1 have ', len(img1_arr), 'ct data')
    print('img2 have ', len(img2_arr), 'ct data')
    return img1_arr, img2_arr


def read_img(path):
    """
    读取nii图像方法
    :param path: 图像路径
    :return: 一个nibabel格式的图片
    """
    img = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(img)
    return image_array

def save_cube_nii(save_path, img_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('保存目录已创建完毕：', save_path)
    pretrain_file = open(r"E:\Data\ribseg_v2\pretrain.txt", 'r')
    pretrain_val_names = pretrain_file.readlines()
    img_names = []
    for pretrain_val_name in pretrain_val_names:
        pretrain_val_name = pretrain_val_name.strip()+'-image.nii.gz'
        img_names.append(pretrain_val_name)
    # img_names = os.listdir(img_path)
    for name in img_names:
        print(os.path.join(img_path, name))
        cube_folder = name.split('-image.nii.gz')[0]
        if not os.path.exists(os.path.join(save_path, cube_folder)):
            os.makedirs(os.path.join(save_path, cube_folder))
        else:
            continue
        img1_arr, img2_arr = img_preprocess(os.path.join(img_path, name))
        for i in range(len(img1_arr)):
            # 创建保存地址
            cube_pair_path = os.path.join(save_path, cube_folder, str(i))
            if not os.path.exists(cube_pair_path):
                os.makedirs(cube_pair_path)

            cube1 = sitk.GetImageFromArray(img1_arr[i])
            cube2 = sitk.GetImageFromArray(img2_arr[i])
            sitk.WriteImage(cube1, os.path.join(cube_pair_path, 'image.nii.gz'))
            sitk.WriteImage(cube2, os.path.join(cube_pair_path, 'image_shift.nii.gz'))
            print(cube_pair_path, '已完成')

"""
上面部分是预训练数据的预处理和保存
下面部分是训练、验证以及测试数据的预处理与保存
区别在于：预训练数据没有标签且不需要记录cube的位置信息
"""

def img_label_preprocess(img_path, label_path, process=True):
    hu_img = torch.from_numpy(read_img(path=img_path))
    rib_mask = torch.from_numpy(read_img(path=label_path))
    gray_img = set_window(hu_image=hu_img, min_v=-400, max_v=1000)
    bone_binary = bone_thres(gray_img=gray_img, gray_threhold=90)
    bone_binary = connect_check(bone_binary)
    if process:
        del_bone_msk, del_rib_msk, del_rib_img, del_bndbox = del_surplus_mask(bone=bone_binary, rib=rib_mask,
                                                                              image=gray_img)
        del_rib_img *= del_bone_msk
        mask_cubes = crop_cube(del_rib_msk, (QUEUE, HEIGHT, WEIGHT))
    else:
        del_rib_img = gray_img * bone_binary
        mask_cubes = crop_cube(rib_mask, (QUEUE, HEIGHT, WEIGHT))
    img_arr = []
    mask_arr = []
    for (mask_cube, position) in mask_cubes:
        mask_cube = torch.squeeze(torch.Tensor(mask_cube))
        z1, x1, y1, z2, x2, y2 = position
        img_cube = del_rib_img[z1:z2, x1:x2, y1:y2]
        if img_cube.shape == (QUEUE, HEIGHT, WEIGHT) and mask_cube.shape == (QUEUE, HEIGHT, WEIGHT):
            mask_arr.append(mask_cube)
            img_arr.append(img_cube)
    return img_arr, mask_arr

# def img_label_preprocess(img_path, label_path, process=True):
#     hu_img = torch.from_numpy(read_img(path=img_path))
#     rib_mask = torch.from_numpy(read_img(path=label_path))
#     gray_img = set_window(hu_image=hu_img, min_v=-400, max_v=1000)
#     if process:
#         bone_binary = bone_thres(gray_img=gray_img, gray_threhold=90)
#         bone_binary = connect_check(bone_binary)
#         del_bone_msk, del_rib_msk, del_rib_img, del_bndbox = del_surplus_mask(bone=bone_binary, rib=rib_mask,
#                                                                               image=gray_img)
#         del_rib_img *= del_bone_msk
#         mask_cubes = crop_cube(del_rib_msk, (QUEUE, HEIGHT, WEIGHT))
#     else:
#         mask_cubes = crop_cube(gray_img, (QUEUE, HEIGHT, WEIGHT))
#     img_arr = []
#     mask_arr = []
#     for (mask_cube, position) in mask_cubes:
#         mask_cube = torch.squeeze(torch.Tensor(mask_cube))
#         value, num = torch.unique(mask_cube, return_counts=True)
#         if len(num.numpy()) > 1:
#                 z1, x1, y1, z2, x2, y2 = position
#                 # OrthoSlicer3D(hu_img[z1:z2, x1:x2, y1:y2]).show()
#                 img_cube = del_rib_img[z1:z2, x1:x2, y1:y2]
#                 if img_cube.shape == (QUEUE, HEIGHT, WEIGHT) and mask_cube.shape == (QUEUE, HEIGHT, WEIGHT):

#                     mask_arr.append(mask_cube)
#                     img_arr.append(img_cube)
#     return img_arr, mask_arr



# def save_cube_label_nii(save_path, img_path, label_path, process):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         print('保存目录已创建完毕：', save_path)
#     img_names = os.listdir(img_path)
#     # img_names = ['RibFrac10-image.nii.gz', 'RibFrac102-image.nii.gz', 'RibFrac104-image.nii.gz', 'RibFrac107-image.nii.gz', 'RibFrac112-image.nii.gz', 'RibFrac114-image.nii.gz', 'RibFrac117-image.nii.gz', 'RibFrac118-image.nii.gz', 'RibFrac121-image.nii.gz', 'RibFrac122-image.nii.gz', 'RibFrac123-image.nii.gz', 'RibFrac126-image.nii.gz', 'RibFrac129-image.nii.gz', 'RibFrac131-image.nii.gz', 'RibFrac141-image.nii.gz', 'RibFrac146-image.nii.gz', 'RibFrac152-image.nii.gz', 'RibFrac153-image.nii.gz', 'RibFrac154-image.nii.gz', 'RibFrac160-image.nii.gz', 'RibFrac163-image.nii.gz', 'RibFrac167-image.nii.gz', 'RibFrac170-image.nii.gz', 'RibFrac172-image.nii.gz', 'RibFrac174-image.nii.gz', 'RibFrac177-image.nii.gz', 'RibFrac180-image.nii.gz', 'RibFrac181-image.nii.gz', 'RibFrac193-image.nii.gz', 'RibFrac2-image.nii.gz', 'RibFrac203-image.nii.gz', 'RibFrac204-image.nii.gz', 'RibFrac206-image.nii.gz', 'RibFrac211-image.nii.gz', 'RibFrac212-image.nii.gz', 'RibFrac215-image.nii.gz', 'RibFrac231-image.nii.gz', 'RibFrac239-image.nii.gz', 'RibFrac245-image.nii.gz', 'RibFrac249-image.nii.gz', 'RibFrac252-image.nii.gz', 'RibFrac255-image.nii.gz', 'RibFrac258-image.nii.gz', 'RibFrac263-image.nii.gz', 'RibFrac265-image.nii.gz', 'RibFrac267-image.nii.gz', 'RibFrac270-image.nii.gz', 'RibFrac271-image.nii.gz', 'RibFrac274-image.nii.gz', 'RibFrac275-image.nii.gz', 'RibFrac277-image.nii.gz', 'RibFrac278-image.nii.gz', 'RibFrac279-image.nii.gz', 'RibFrac282-image.nii.gz', 'RibFrac286-image.nii.gz', 'RibFrac289-image.nii.gz', 'RibFrac294-image.nii.gz', 'RibFrac298-image.nii.gz', 'RibFrac299-image.nii.gz', 'RibFrac3-image.nii.gz', 'RibFrac30-image.nii.gz', 'RibFrac301-image.nii.gz', 'RibFrac305-image.nii.gz', 'RibFrac308-image.nii.gz', 'RibFrac31-image.nii.gz', 'RibFrac314-image.nii.gz', 'RibFrac315-image.nii.gz', 'RibFrac316-image.nii.gz', 'RibFrac319-image.nii.gz', 'RibFrac325-image.nii.gz', 'RibFrac33-image.nii.gz', 'RibFrac330-image.nii.gz', 'RibFrac331-image.nii.gz', 'RibFrac336-image.nii.gz', 'RibFrac338-image.nii.gz', 'RibFrac34-image.nii.gz', 'RibFrac342-image.nii.gz', 'RibFrac352-image.nii.gz', 'RibFrac353-image.nii.gz', 'RibFrac354-image.nii.gz', 'RibFrac356-image.nii.gz', 'RibFrac358-image.nii.gz', 'RibFrac359-image.nii.gz', 'RibFrac36-image.nii.gz', 'RibFrac361-image.nii.gz', 'RibFrac362-image.nii.gz', 'RibFrac368-image.nii.gz', 'RibFrac37-image.nii.gz', 'RibFrac373-image.nii.gz', 'RibFrac38-image.nii.gz', 'RibFrac383-image.nii.gz', 'RibFrac384-image.nii.gz', 'RibFrac389-image.nii.gz', 'RibFrac392-image.nii.gz', 'RibFrac394-image.nii.gz', 'RibFrac398-image.nii.gz', 'RibFrac401-image.nii.gz', 'RibFrac402-image.nii.gz', 'RibFrac403-image.nii.gz', 'RibFrac404-image.nii.gz', 'RibFrac41-image.nii.gz', 'RibFrac416-image.nii.gz', 'RibFrac420-image.nii.gz', 'RibFrac45-image.nii.gz', 'RibFrac46-image.nii.gz', 'RibFrac48-image.nii.gz', 'RibFrac55-image.nii.gz', 'RibFrac65-image.nii.gz', 'RibFrac66-image.nii.gz', 'RibFrac68-image.nii.gz', 'RibFrac7-image.nii.gz', 'RibFrac70-image.nii.gz', 'RibFrac71-image.nii.gz', 'RibFrac77-image.nii.gz', 'RibFrac79-image.nii.gz', 'RibFrac81-image.nii.gz', 'RibFrac83-image.nii.gz', 'RibFrac89-image.nii.gz', 'RibFrac90-image.nii.gz', 'RibFrac96-image.nii.gz']
#     count = 0
#     for name in tqdm(img_names):

#         print(os.path.join(img_path, name))
#         mask_name = name.split('-image.nii.gz')[0] + '-rib-seg.nii.gz'
#         # mask_name = name.split('_IMG.nii.gz')[0] + '_MSK.nii.gz'
#         print(mask_name)

#         # 如果这个数据有标注，就当做训练数据
#         if os.path.exists(os.path.join(label_path, mask_name)):
#             # 创建保存地址
#             cube_folder = name.split('-image.nii.gz')[0]
#             # cube_folder = name.split('_IMG.nii.gz')[0]
#             if not os.path.exists(os.path.join(save_path, cube_folder)):
#                 os.makedirs(os.path.join(save_path, cube_folder))
#             else:
#                 print('{}已处理'.format(cube_folder))
#                 count += 1
#                 print('共处理{}个CT图片'.format(count))
#                 continue
#             img_arr, mask_arr = img_label_preprocess(img_path=os.path.join(img_path, name), label_path=os.path.join(label_path, mask_name), process=process)

#             for i in range(len(img_arr)):
#                 cube_mask_path = os.path.join(save_path, cube_folder, str(i))
#                 if not os.path.exists(cube_mask_path):
#                     os.makedirs(cube_mask_path)
#                 # img = sitk.GetImageFromArray(img_arr[i])
#                 # mask = sitk.GetImageFromArray(mask_arr[i])
#                 # sitk.WriteImage(img, os.path.join(cube_mask_path, 'image.nii.gz'))
#                 # sitk.WriteImage(mask, os.path.join(cube_mask_path, 'mask.nii.gz'))
#                 np.savez(os.path.join(cube_mask_path,'img.npz'),img_arr[i])
#                 np.savez(os.path.join(cube_mask_path,'msk.npz'),mask_arr[i])

#                 print(cube_mask_path, '已完成')
#             count += 1
#         print('共处理{}个CT图片'.format(count))

def save_cube_label_nii(save_path, img_path, label_path, process_num, process):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('保存目录已创建完毕：', save_path)
    img_names = os.listdir(img_path)
    count = 0
    for name in tqdm(img_names):

        print(os.path.join(img_path, name))
        # mask_name = name.split('_IMG.nii.gz')[0] + '_MSK.nii.gz'
        mask_name = name.split('-image.nii.gz')[0] + '-rib-seg.nii.gz'

        print(mask_name)

        # 如果这个数据有标注，就当做训练数据
        if os.path.exists(os.path.join(label_path, mask_name)):
            # 创建保存地址
            # cube_folder = name.split('_IMG.nii.gz')[0]
            cube_folder = name.split('-image.nii.gz')[0]

            if not os.path.exists(os.path.join(save_path, cube_folder)):
                os.makedirs(os.path.join(save_path, cube_folder))
            else:
                print('{}已处理'.format(cube_folder))
                count += 1
                print('共处理{}个CT图片'.format(count))
                continue
            img_arr, mask_arr = img_label_preprocess(img_path=os.path.join(img_path, name), label_path=os.path.join(label_path, mask_name), process=process)
            process_num -= len(img_arr)
            if process_num >= 0:
                for i in range(len(img_arr)):
                    cube_mask_path = os.path.join(save_path, cube_folder, str(i))
                    if not os.path.exists(cube_mask_path):
                        os.makedirs(cube_mask_path)
                    np.savez(os.path.join(cube_mask_path, "img.npz"), img_arr[i])
                    np.savez(os.path.join(cube_mask_path, "msk.npz"), mask_arr[i])
                    print(cube_mask_path, '已完成')
            else: 
                return
            print('还需要处理{}个cube'.format(process_num))
            count += 1
        print('共处理{}个CT图片'.format(count))


def del_surplus_mask(bone, rib, image):
    """
    提取肺实质的边界框
    :param bone: 骨分割mask
    :param rib: 肋骨分割标注mask
    :param image:  影像
    :return mask_del: 删减无关区域后的骨分割mask
    :return rib_del: 删减无关区域的后的肋骨分割标注mask
    :return image_del: 删减无关无区域骺的image
    :return 删减的边界框
    """
    zs, ys, xs = np.where(bone)
    zmin, zmax = np.min(zs), np.max(zs)
    ymin, ymax = np.min(ys), np.max(ys)
    xmin, xmax = np.min(xs), np.max(xs)

    bone_del = bone[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    rib_del = rib[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    image_del = image[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]

    return bone_del, rib_del, image_del.astype(np.int32), (zmin, ymin, xmin, zmax, ymax, xmax)


def test_process(imgs_path, labels_path, pre_handle=True):
    test_epoch=99
    peth=0.6
    test_type = "ALL_36000"
    names = os.listdir(imgs_path)
    for img_name in names:
        # if "-image.nii.gz" not in img_name:
        #     img_name = img_name+"-image.nii.gz"
        # mask_name = img_name.split('_IMG.nii.gz')[0]+'_MSK.nii.gz'
        mask_name = img_name.split('-image.nii.gz')[0]+'-rib-seg.nii.gz'
        img_path = os.path.join(imgs_path, img_name)
        mask_path = os.path.join(labels_path, mask_name)
        print(img_path)
        img = torch.from_numpy(read_img(path=img_path))
        mask = read_img(path=mask_path)
        img = set_window(hu_image=img, min_v=-400, max_v=1000)
        if pre_handle:
            bone_binary = bone_thres(gray_img=img, gray_threhold=90)
            bone_binary = connect_check(bone_binary)
            del_bone_msk, del_rib_msk, del_rib_img, del_bndbox = del_surplus_mask(bone=bone_binary, rib=mask, image=img)
            del_rib_img *= del_bone_msk
            img = del_rib_img
            mask = del_rib_msk

        shape = (mask.shape[0] + 64, mask.shape[1] + 64, mask.shape[2] + 64)
        aug_msk = np.zeros(shape)
        aug_msk[32:shape[0] - 32, 32:shape[1] - 32, 32:shape[2] - 32] = mask
        shape = (img.shape[0] + 64, img.shape[1] + 64, img.shape[2] + 64)
        aug_img = np.zeros(shape)
        aug_img[32:shape[0] - 32, 32:shape[1] - 32, 32:shape[2] - 32] = img

        aug_msk = (aug_msk > 0).astype(np.int8)
        mask_img = sitk.GetImageFromArray(aug_msk)
        img = sitk.GetImageFromArray(aug_img)
        # save_path = "/me4012/RibSegV2/data/yyh_2025/PEth0901/prediction_{}/source_peth{}/".format(test_epoch, peth) +img_name.split('-image.nii.gz')[0]
        save_path = "/me4012/RibSegV2/data/yyh_2025/PEth0901/prediction_{}/no{}/".format(test_epoch, test_type) +img_name.split('-image.nii.gz')[0]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sitk.WriteImage(mask_img, os.path.join(save_path, 'mask.nii.gz'))
        sitk.WriteImage(img, os.path.join(save_path, 'image.nii.gz'))
        cubes = crop_cube(aug_img, (QUEUE, HEIGHT, WEIGHT))
        # yield cubes, aug_img.shape, img_name.split('_IMG.nii.gz')[0], del_bndbox, img_shape
        yield cubes, aug_img.shape, img_name.split('-image.nii.gz')[0], torch.from_numpy(aug_msk)


def crop_cube_no_overlap(gray_img_arr, cube_size):
    """
    滑窗式裁剪cube
    :param gray_img_arr: 待裁剪的影像数组
    :param cube_size: 参数如（64， 64， 64）, 裁剪的cube的尺寸
    :return: cube, (zmin, ymin, xmin, zmax, ymax, xmax)
    """
    depth, height, width = gray_img_arr.shape
    cube_z, cube_y, cube_x = cube_size
    for z in range(0, depth, int(cube_z)):
        for y in range(0, height, int(cube_y)):
            for x in range(0, width, int(cube_x)):
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

def test_process_no_multiView(path):
    imgs_path = os.path.join(path, 'series')
    names = os.listdir(imgs_path)
    # names = open("E:\Data\\rib_segmentation_CT\\test.txt")
    for img_name in names:
        # img_name = 'FTRIB' + name.split('\n')[0] + '_img.nii.gz'
        mask_name = img_name.split('_IMG.nii.gz')[0]+'_MSK.nii.gz'
        # mask_name = name+'_msk.nii.gz'
        img_path = os.path.join(imgs_path, img_name)
        mask_path = os.path.join(path, 'mask', mask_name)
        print(img_path)
        hu_img = torch.from_numpy(read_img(path=img_path))
        rib_mask = read_img(path=mask_path)
        gray_img = set_window(hu_image=hu_img, min_v=-400, max_v=1000)
        bone_binary = bone_thres(gray_img=gray_img, gray_threhold=90)
        bone_binary = connect_check(bone_binary)
        del_bone_msk, del_rib_msk, del_rib_img, del_bndbox = del_surplus_mask(bone=bone_binary, rib=rib_mask, image=gray_img)
        del_rib_img *= del_bone_msk

        mask = sitk.GetImageFromArray(del_rib_msk)
        img = sitk.GetImageFromArray(del_rib_img)
        save_path = "E:\Data\prediction\decoder_200_v0.5/"+img_name.split('_IMG.nii.gz')[0]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        sitk.WriteImage(mask, os.path.join(save_path, 'mask.nii.gz'))
        sitk.WriteImage(img, os.path.join(save_path, 'image.nii.gz'))
        cubes = crop_cube_no_overlap(del_rib_img, (QUEUE, HEIGHT, WEIGHT))
        yield cubes, del_rib_img.shape, img_name.split('_IMG.nii.gz')[0]

def del_zero():
    data_path = '/data/yyh/Processed Data/ribSegV2/ribSegv2_val'
    datas = os.listdir(data_path)
    count = 0
    for data in datas:
        cube_folder = os.path.join(data_path, data)
        cubes = os.listdir(cube_folder)
        for cube in cubes:
            cube_path = os.path.join(cube_folder, cube)
            label_path = os.path.join(cube_path, 'mask.nii.gz')
            label = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float64))
            num = torch.unique(label)
            if len(num) == 1 and num[0] == 0:
                print(torch.unique(label, return_counts=True))
                shutil.rmtree(cube_path)
                count += 1
                print('{}已被删除，已删除{}个cube'.format(cube_path, count))

def post_process(data):
    # 应用连通区域标记
    labels, num_features = ndimage.label(data)

    # 计算每个连通区域的大小
    sizes = ndimage.sum_labels(data, labels, range(1, num_features + 1))

    # 设定阈值，例如只保留大小超过 100 的连通域
    threshold = 5000

    # 创建一个新的标签数组，用于存储较大的连通域
    large_labels = np.zeros_like(labels)

    # 遍历所有连通区域，只保留大小超过阈值的区域
    for label in range(1, num_features + 1):
        if sizes[label - 1] > threshold:
            large_labels[labels == label] = label

        # 将较大的连通域重新标记为 1, 2, ..., 以便后续处理
    large_labels, num_large_features = ndimage.label(large_labels)
    large_regions = torch.from_numpy((large_labels > 0).astype(np.int8))
    return large_regions


def pre_process(img, mask=None, process=True):
    gray_img = set_window(hu_image=img, min_v=-400, max_v=1000)
    if process:
        bone_binary = bone_thres(gray_img=gray_img, gray_threhold=90)
        bone_binary = connect_check(bone_binary)
        if mask is None:
            del_bone_msk, del_rib_img, del_bndbox = del_surplus_noMask(bone=bone_binary, image=gray_img)
        else:
            del_bone_msk, del_rib_msk, del_rib_img, del_bndbox = del_surplus_mask(bone=bone_binary, rib=mask, image=gray_img)
            mask = del_rib_msk * del_bone_msk
        img = del_rib_img * del_bone_msk
    else:
        img = gray_img
    cubes = crop_cube(img, (64, 64, 64))
    if mask is None:
        return cubes, img
    else:
        return cubes, img, mask

def collect_data_by_target_group(root_dir,target_text):

    """
    遍历 root_dir，只返回主文件夹名在target_text内的分组的所有二级子文件夹
    返回格式: dict {主文件夹名: [子文件夹路径1, 子文件夹路径2, ...]}
    """
    with open(target_text, 'r', encoding='utf-8') as f:
        target_folders = set(line.strip() for line in f if line.strip())  # 去掉空行和换行符
    grouped_data = defaultdict(list)
    for main_folder in os.listdir(root_dir):
        if main_folder in target_folders:
            main_path = os.path.join(root_dir, main_folder)
            if os.path.isdir(main_path):
                for subfolder in os.listdir(main_path):
                    subfolder_path = os.path.join(main_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        grouped_data[main_folder].append(subfolder_path)
    return grouped_data

def count_subfolders(folder_path):
    """
    统计某个主文件夹下有多少个子文件夹（即数据条数）
    """
    return sum(
        os.path.isdir(os.path.join(folder_path, sub)) 
        for sub in os.listdir(folder_path)
    )

def collect_random_groups(root_dir, limit=50000, seed=42):
    """
    随机选择主文件夹，每次选择一个就复制该主文件夹的全部内容，
    在复制下一个主文件夹之前，如果总数已经 > limit，则停止。
    """
    # 获取所有主文件夹
    main_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    random.seed(seed)
    random.shuffle(main_folders)  # 随机打乱

    selected_groups = []  # 最终选择的主文件夹
    total_count = 0

    for group in main_folders:
        group_path = os.path.join(root_dir, group)
        group_count = count_subfolders(group_path)

        # 检查是否已经超过限制，如果是则停止，不再选择当前组
        if total_count > limit:
            break

        # 选择当前组
        selected_groups.append(group_path)
        total_count += group_count
        print(f"选择主文件夹: {group} (含 {group_count} 条数据)，当前累计: {total_count}")

    return selected_groups, total_count


def allocate_samples(grouped_data, target_total):
    """
    根据每组数据量，计算分配的采样数
    """
    group_sizes = {k: len(v) for k, v in grouped_data.items()}
    total_count = sum(group_sizes.values())
    
    # 初步分配
    allocation = {
        k: round(size / total_count * target_total)
        for k, size in group_sizes.items()
    }

    # 调整分配，保证最终和为 target_total
    diff = target_total - sum(allocation.values())
    if diff != 0:
        # 按每组数据量大小排序，逐一增减
        sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=(diff > 0))
        for group, _ in sorted_groups:
            allocation[group] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            if diff == 0:
                break

    return allocation


def sample_folders_by_group(grouped_data, allocation):
    """
    在每个主文件夹中随机采样 allocation[group] 个数据
    """
    sampled_folders = []
    for group, folders in grouped_data.items():
        num_to_sample = allocation[group]
        if num_to_sample > len(folders):
            raise ValueError(f"分组 {group} 中数据不足: 需要采样 {num_to_sample} 个，但只有 {len(folders)} 个")
        sampled = random.sample(folders, num_to_sample)
        sampled_folders.extend(sampled)
    return sampled_folders


def copy_with_structure(sampled_folders, root_dir, new_root_dir):
    """
    将采样到的文件夹复制到新目录，保持目录层级
    """
    for folder in tqdm(sampled_folders, desc="Copying data"):
        # 保留相对路径
        relative_path = os.path.relpath(folder, root_dir)
        target_path = os.path.join(new_root_dir, relative_path)

        # 创建目标路径
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 拷贝整个子文件夹
        shutil.copytree(folder, target_path)


def create_no_EGPA_dataset():
    # ===== 用户自定义 =====
    root_dir = "/me4012/RibSegV2/data/yyh_2025/PEth0901/RibSegV2/train/"    # 原始数据目录
    new_root_dir = "/me4012/RibSegV2/data/yyh_2025/PEth0901/Processed Data/no_EGPA_source_peth0.6/"     # 输出采样目录
    target_text = "/me4012/RibSegV2/data/yyh_2025/PEth0901/Processed Data/querysample_source_peth0.6/selected_sample.txt"
    target_total = 50000                   # 目标采样总数
    random.seed(42)                        # 固定随机种子，保证结果可复现

    # ===== 1. 收集数据 =====
    print("正在收集数据分组...")
    # grouped_data = collect_data_by_group(root_dir)
    grouped_data = collect_data_by_target_group(root_dir,target_text)
    total_data_count = sum(len(v) for v in grouped_data.values())
    print(f"总共有 {len(grouped_data)} 个主文件夹，共 {total_data_count} 条数据")

    # ===== 2. 分配采样数 =====
    print("正在计算采样分配...")
    allocation = allocate_samples(grouped_data, target_total)
    for group, num in allocation.items():
        print(f"  {group}: 采样 {num} 条 (共 {len(grouped_data[group])} 条)")

    # ===== 3. 按分组采样 =====
    print("正在随机采样...")
    sampled_folders = sample_folders_by_group(grouped_data, allocation)
    print(f"采样完成，共 {len(sampled_folders)} 条数据")

    # ===== 4. 复制采样结果 =====
    print("正在复制采样数据到新目录...")
    copy_with_structure(sampled_folders, root_dir, new_root_dir)
    print("复制完成！")

def create_no_all_dataset():
        # ===== 用户自定义 =====
    root_dir = "/me4012/RibSegV2/data/yyh_2025/PEth0901/RibSegV2/train/" # 原始数据目录
    new_root_dir = "/me4012/RibSegV2/data/yyh_2025/PEth0901/Processed Data/no_all_source_peth0.6_36000/"   # 采集后保存目录
    limit = 36000                         # 采集目标总数
    seed = 42 #第一三次使用
    # seed = 915                             # 第二次使用

    # ===== 1. 随机采集 =====
    print(f"正在随机选择主文件夹，直到数据总量超过 {limit} ...")
    selected_groups, final_count = collect_random_groups(root_dir, limit, seed)
    print(f"\n最终选择了 {len(selected_groups)} 个主文件夹，总数据量: {final_count}")

    # ===== 2. 复制结果 =====
    print("正在复制数据到新目录...")
    copy_with_structure(selected_groups, root_dir, new_root_dir)
    print("复制完成！")

if __name__ == '__main__':
    # create_no_all_dataset()
    # save_cube_nii(save_path=r"E:\Data\processed_data\ribSegv2_pretrain", img_path=r"E:\Data\ribseg_v2\ribfrac-train-images")
    # list = ['train', 'val', 'test']
    # for list_1 in list:
    #     save_path = "/data2/cly/ribseg_dataset/process_dataset/train"
    #     img_path = "/data2/cly/ribseg_dataset/dataset/v0.5/series"
    #     # img_path = "/me4012/RibSegV2/data/ribfrac-val-images"
    #     label_path = "/data2/cly/ribseg_dataset/dataset/v0.5/mask"
    #     save_cube_label_nii(save_path=save_path,
    #                         img_path=img_path,
    #                         label_path=label_path,
    #                         process=True)
    # save_path = "/me4012/RibSegV2/data/yyh_2025/PEth0901/RibSegV2/train"
    save_path = "/me4012/RibSegV2/data/yyh_2025/PEth0901/Processed Data/no_all_source_peth0.6_36000/"
    img_path = "/me4012/RibSegV2/data/ribfrac-train-images/"
    # img_path = "/me4012/RibSegV2/data/ribfrac-val-images"
    
    label_path = "/me4012/RibSegV2/data/label/seg/"
    save_cube_label_nii(save_path=save_path,
                        img_path=img_path,
                        label_path=label_path,
                        process_num=36000,
                        process=False)
    # del_zero()