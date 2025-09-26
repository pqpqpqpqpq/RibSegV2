import os
import cv2
import numpy as np

import SimpleITK as sitk
# from skimage import measure


def set_window(hu_image, min_v=-400, max_v=1000):
    norm_img = (hu_image - min_v) / (max_v - min_v)
    norm_img = np.clip(norm_img, 0, 1)
    gray_img = (norm_img * 255).numpy().astype(np.uint8)

    return gray_img


def bone_thres(gray_img, gray_threhold=90):
    """
    按照灰度阈值进行骨分割
    :param gray_img: 灰度图像数组 [0,255] uint8
    :param gray_threhold: 灰度分割阈值
    :return : 阈值分割后的二值图像numpy数组 [0,255] uint8
    """
    # 高斯平滑
    filter_img = [cv2.GaussianBlur(gray_img[s], (5, 5), 0) for s in range(gray_img.shape[0])]
    # 阈值分割 t=90
    binary_img = [cv2.threshold(filter_img[s], gray_threhold, 255, cv2.THRESH_BINARY)[1] for s in range(gray_img.shape[0])]
    # 中值滤波，去除小噪点，减少标记连通域的计算
    filter_binary = [cv2.medianBlur(binary_img[s], 3) for s in range(gray_img.shape[0])]
    # 形态学开闭运算，断开骨骼和金属板，填充内部孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_binary = [cv2.morphologyEx(filter_binary[s], cv2.MORPH_CLOSE, kernel) for s in range(gray_img.shape[0])]

    return np.array(open_binary)


def fill(img_slice):
    """
    水漫填充,类型为np.uint8，需要先转换为[0, 255]二值化
    :param im_slice: 单层影像的二值化mask
    :return:
    """
    im_in = img_slice * 255
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)  # 填充 255
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # ---> Modified 1
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv  # [0, 255]二值化

    return im_out / 255


def fill3d(nodule_msk):
    new_msk = np.zeros(nodule_msk.shape, dtype=np.int32)
    for z in range(nodule_msk.shape[0]):
        tmp = nodule_msk[z, ...].astype(np.uint8)
        filled_tmp = fill(tmp.copy()).astype(np.int32)
        new_msk[z, ...] = filled_tmp

    return new_msk


def connect_check(binary_img):
    '''
    #三维连通域标记，选择保留的连通区域，生成mask
    :param binary_img: 阈值分割后的二值图像 [0,255] uint8
    :return mask: [0,1] uint8
    '''
    binary = sitk.GetImageFromArray(binary_img)
    connectTool = sitk.ConnectedComponentImageFilter()
    connectTool.SetFullyConnected(True)
    connected = connectTool.Execute(binary)
    conneted_img = sitk.GetArrayFromImage(connected)
    label_num = connectTool.GetObjectCount()
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected)

    num_list = [i for i in range(1, label_num + 1)]  # GetNumberOfPixels计算不包含背景像素label
    area_list = []  # num_list中全部的面积
    label_list = []  # num_list中部分保留的label

    for l in num_list:
        area = stats.GetNumberOfPixels(l)
        area_list.append(stats.GetNumberOfPixels(l))
        if area > 1000:  # 去除小连通域，减少排序数量，加速
            # print(l, area)
            label_list.append(l)
    label_list_sorted = sorted(label_list, key=lambda x: area_list[x - 1], reverse=True)

    mask = np.zeros(conneted_img.shape).astype('uint8')
    for label in label_list_sorted[:1]:  # 选择保留的连通域
        mask[conneted_img == label] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expand_mask = np.array([cv2.morphologyEx(mask[s], cv2.MORPH_DILATE, kernel) for s in range(mask.shape[0])])
    filled_mask = fill3d(expand_mask)

    return filled_mask


def del_surplus(bone, rib, image):
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


def preprocess_ribs(hu_img, rib_msk):
    """
    肋骨分割预处理，分割胸部CT平扫的骨组织
    :return del_rib_img: 删除周边无骨组织区域的灰度影像
    :return del_rib_msk: 删除周边无骨组织区域的mask
    :return del_bndbox: (zmin, ymin, xmin, zmax, ymax, xmax)
    """
    rib_label_msk = measure.label(rib_msk)
    label_ids, label_counts = np.unique(rib_label_msk, return_counts=True)
    for i in range(label_ids.shape[0]):
        label_id = label_ids[i]
        if label_id == 0:
            continue
        label_count = label_counts[i]
        if label_count <= 50:
            rib_msk[rib_label_msk == label_id] = 0
    bone_gray_image = set_window(hu_image=hu_img, min_v=-400, max_v=1000)  # 骨窗转换
    bone_mask = bone_thres(gray_img=bone_gray_image, gray_threhold=90)   # 灰度阈值分割
    bone_mask = connect_check(bone_mask)   # 计算连通区域并填充
    # print(np.where(bone_mask))
    del_bone_msk, del_rib_msk, del_rib_img, del_bndbox = del_surplus(bone=bone_mask, rib=rib_msk, image=bone_gray_image)
    # 删除外围区域
    del_rib_img *= del_bone_msk   # 实现影像分割，背景灰度为0

    return del_rib_img, del_rib_msk, del_bone_msk, del_bndbox


def plot_arr(save_path, img, bone, rib):
    for z in range(img.shape[0]):
        tmp = np.zeros((img.shape[1], img.shape[2]*3))
        tmp_img = img[z]
        tmp_bone = (bone[z] * 255).astype(np.uint8)
        tmp_rib = (rib[z] * 255).astype(np.uint8)
        tmp[:, :tmp_img.shape[1]] = tmp_img
        tmp[:, tmp_img.shape[1]:(2 * tmp_img.shape[1])] = tmp_bone
        tmp[:, (2 * tmp_img.shape[1]):(3 * tmp_img.shape[1])] = tmp_rib
        cv2.imwrite(os.path.join(save_path, "{}.png".format(str(z).zfill(3))), tmp)


if __name__ == "__main__":
    import shutil

    PATH = "/data/feituai/rib_segmentation_CT/v0.3.220808"
    IMG_PATH = os.path.join(PATH, "rib_seg_need_label")
    MSK_PATH = os.path.join(PATH, "manual_label_final")
    PIC_PATH = os.path.join(PATH, "pictures")
    PREPROCESS_PATH = os.path.join(PATH, "preprocess")
    if not os.path.exists(PREPROCESS_PATH):
        os.makedirs(PREPROCESS_PATH)
    p_list = os.listdir(IMG_PATH)
    # p_list = ["FTRIB0124"]
    for pname in p_list:
        img_file = os.path.join(IMG_PATH, pname, "{}_img.nii.gz".format(pname))
        msk_file = os.path.join(MSK_PATH, "{}_msk.nii.gz".format(pname))
        if os.path.exists(msk_file):
            img_itk = sitk.ReadImage(img_file)
            hu_img = sitk.GetArrayFromImage(img_itk)
            msk_itk = sitk.ReadImage(msk_file)
            rib_msk = sitk.GetArrayFromImage(msk_itk)
            if hu_img.shape[0] >= 100:
                print(pname)
                shutil.copyfile(img_file, os.path.join(PATH, "series", "{}_img.nii.gz".format(pname)))
                del_rib_img, del_rib_msk, del_bone_msk, del_bndbox = preprocess_ribs(hu_img=hu_img, rib_msk=rib_msk)
                save_path = os.path.join(PIC_PATH, pname)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plot_arr(save_path, img=del_rib_img, bone=del_bone_msk, rib=del_rib_msk)
                np.savez_compressed(os.path.join(PREPROCESS_PATH, "{}.npz".format(pname)),
                         img=del_rib_img, msk=del_rib_msk)