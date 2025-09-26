import os
import cv2

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from skimage import measure

from .read_scan import read_dcms
from .utils import segment_lung_mask_slice_fast, largest_label_image, extend_mask
from .utils import fill3d


class ChestCTImage(object):
    def __init__(self, series_path):
        self.hu_image, self.old_spacing = read_dcms(series_path)
        self.scan_shape = self.hu_image.shape
        self.lung_pad_value = 170

    @staticmethod
    def setWindow(hu_image, mode="minmax", v1=-1200.0, v2=600.0):
        """
        调整窗宽窗位utils.py
        :param hu_image: HU原始影像
        :param mode: minmax 或 wwwc
        :param v1:  若为wwwc，则v1是wc
        :param v2:  若为wwwc，则v2是ww
        :return gray_img: 灰度影像, ndarray
        """
        assert mode in ["minmax", "wwwc"]
        if mode == "minmax":
            min_v = min(v1, v2)
            max_v = max(v1, v2)
        else:
            wc = v1
            ww = v2
            min_v = wc - ww / 2
            max_v = wc + ww / 2

        norm_img = (hu_image - min_v) / (max_v - min_v)
        norm_img = np.clip(norm_img, 0, 1)
        gray_img = (norm_img * 255).astype(np.uint8)

        return gray_img

    def segment_lung_pattern(self, hu_threshold=-600,  fill_lung_structures=True, dilation=True):
        """
        肺实质分割
        :param hu_threshold: 分割阈值
        :param fill_lung_structures: 填充
        :param dilation:
        :return:
        """
        # not actually binary, but 1 and 2.
        # 0 is treated as background, which we do not want
        binary_image = np.array(self.hu_image > hu_threshold, dtype=np.int8) + 1
        # get through the light line at the bottom
        binary_image = np.pad(binary_image, ((2, 0), (0, 0), (0, 0)), 'constant',
                              constant_values=((2, 2), (2, 2), (2, 2)))
        binary_image[0, :, :] = 1
        binary_image[:, 0, :] = 1
        binary_image[:, -1, :] = 1
        # 获得阈值图像
        labels = measure.label(binary_image, connectivity=1)
        # label()函数标记连通区域
        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_label = labels[0, 0, 0]
        # Fill the air around the person
        binary_image[labels == background_label] = 2
        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max, _ = largest_label_image(labeling, bg=labeling[0, 0])
                if l_max is not None:  # This slice contains some lung
                    binary_image[i][np.logical_and(labeling != l_max, labeling != labeling[0, 0])] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1
        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max, maxcount = largest_label_image(labels, bg=0)
        if l_max is not None:  # There are air pockets
            background_label = labels[0, 0, 0]
            min_label = labels.min()
            _, labelcounts = np.unique(labels, return_counts=True)
            maxlabel = labels.max()
            label_coords = np.where(labels != background_label)
            for lci in range(len(label_coords[0])):
                label = labels[label_coords[0][lci]][label_coords[1][lci]][label_coords[2][lci]] - min_label
                if labelcounts[label] < maxcount / 4:
                    binary_image[label_coords[0][lci]][label_coords[1][lci]][label_coords[2][lci]] = 0

        binary_image = np.delete(binary_image, [0, 1], axis=0)
        if maxcount < self.hu_image.size * 0.001:
            binary_image = np.zeros_like(self.hu_image, dtype=np.int8)
            for z in range(len(binary_image)):
                binary_image[z] = segment_lung_mask_slice_fast(self.hu_image[z], hu_threshold)
            if (binary_image > 0).sum() < self.hu_image.size * 0.001:
                return None
        if np.count_nonzero(binary_image[:, -1, :]) > 0:
            # wrong segmentation, which need correction
            labeling = measure.label(binary_image, background=0)
            for l in np.unique(labeling[:, -1, :]):
                if l != 0:
                    binary_image[labeling == l] = 0
        if dilation:
            binary_image = extend_mask(binary_image, ball_size=3, iterations=6)

        return binary_image

    @staticmethod
    def bone_thres(gray_img, gray_threhold=90):
        """
        按照灰度阈值进行骨分割
       :param gray_img: 灰度图像数组 [0,255] uint8
       :param gray_threhold: 灰度分割阈值
       :return : 阈值分割后的二值图像numpy数组 [0,255] uint8
       """
        # 高斯平滑
        filter_img = [cv2.GaussianBlur(gray_img[s], (5, 5), 0) for s in range(gray_img.shape[0])]
        # slider_show(filter_img)

        # 阈值分割 t=90
        binary_img = [cv2.threshold(filter_img[s], gray_threhold, 255, cv2.THRESH_BINARY)[1] for s in range(gray_img.shape[0])]
        # slider_show(binary_img)

        # 中值滤波，去除小噪点，减少标记连通域的计算
        filter_binary = [cv2.medianBlur(binary_img[s], 3) for s in range(gray_img.shape[0])]
        # slider_show(filter_binary)

        # 形态学开闭运算，断开骨骼和金属板，填充内部孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_binary = [cv2.morphologyEx(filter_binary[s], cv2.MORPH_CLOSE, kernel) for s in range(gray_img.shape[0])]
        # slider_show(open_binary)
        # plot_ct_scan(open_binary[:100])

        return np.array(open_binary)

    @staticmethod
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

    @staticmethod
    def del_surplus(mask, image):
        """
        提取肺实质的边界框
        :param mask: mask
        :param image:  影像
        :return mask_del: 删减无关区域后的mask
        :return image_del: 删减无关无区域骺的image
        :return 删减的边界框
        """
        zs, ys, xs = np.where(mask)
        zmin, zmax = np.min(zs), np.max(zs)
        ymin, ymax = np.min(ys), np.max(ys)
        xmin, xmax = np.min(xs), np.max(xs)

        mask_del = mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
        image_del = image[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]

        return mask_del, image_del.astype(np.int32), (zmin, ymin, xmin, zmax, ymax, xmax)

    def resample(self, image, new_spacing=None):
        """
        重采样，通过插值函数将层厚处理的更薄
        :param image:  gray image, 灰度影像
        :param new_spacing:  目标体素间距
        :return resampled_image: 经过重采样的影像
        :return new_spacing: 重采样后的实际体素间距
        """
        if new_spacing is None:
            new_spacing = [1, 1, 1]

        if self.old_spacing is None:
            raise Exception("Old spacing must not be None")

        resize_factor = self.old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = self.old_spacing / real_resize_factor
        resampled_image = np.expand_dims(image, axis=0)
        resampled_image = np.expand_dims(resampled_image, axis=1)
        input_tsr = torch.from_numpy(resampled_image).cuda()
        resized_tsr = F.interpolate(input_tsr, scale_factor=tuple(real_resize_factor), mode='nearest')
        resampled_image = resized_tsr.squeeze().cpu().data.numpy()

        return resampled_image, new_spacing

    def preprocessLung(self):
        """
        肺实质预处理
        :return resampled_gray_image: 重采样后的肺实质图像
        :return pad_gray_image: 利用肺实质mask进行填充的肺实质图像
        :return new_spacing: ndarray, (z, y, x)
        :return del_bndbox: (zmin, ymin, xmin, zmax, ymax, xmax)
        """
        if self.old_spacing[0] < 2:
            new_spacing = [0.5, 0.5, 0.5]
        else:
            new_spacing = [1, 1, 1]
        lung_pattern_mask = self.segment_lung_pattern(hu_threshold=-600, fill_lung_structures=True, dilation=True)
        del_mask, del_hu_image, del_bndbox = self.del_surplus(lung_pattern_mask, self.hu_image)
        lung_gray_image = self.setWindow(del_hu_image.copy(), mode="minmax", v1=-1200, v2=600)
        pad_gray_image = lung_gray_image * del_mask + self.lung_pad_value * (1 - del_mask).astype(np.uint8)
        resampled_gray_image, new_spacing = self.resample(pad_gray_image.copy(), new_spacing)

        return resampled_gray_image, new_spacing, del_bndbox, lung_pattern_mask, pad_gray_image

    def preprocessRibs(self):
        """
        肋骨分割预处理，分割胸部CT平扫的骨组织
        :return del_rib_img: 删除周边无骨组织区域的灰度影像
        :return del_rib_msk: 删除周边无骨组织区域的mask
        :return del_bndbox: (zmin, ymin, xmin, zmax, ymax, xmax)
        """
        rib_gray_image = self.setWindow(hu_image=self.hu_image, mode="minmax", v1=-400.0, v2=1000.0)  # 骨窗转换
        rib_mask = self.bone_thres(gray_img=rib_gray_image, gray_threhold=90)   # 灰度阈值分割
        rib_mask = self.connect_check(rib_mask)   # 计算连通区域并填充
        del_rib_msk, del_rib_img, del_bndbox = self.del_surplus(mask=rib_mask, image=rib_gray_image)  # 删除外围区域
        del_rib_img *= del_rib_msk   # 实现影像分割，背景灰度为0

        return del_rib_img, del_rib_msk, del_bndbox

