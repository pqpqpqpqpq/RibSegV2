import os

import SimpleITK as sitk
import numpy as np
import pydicom as pdc


def sort_dcms(case_path):
    """
    将文件夹中的序列排序，防止读取影像时层断错乱
    :param case_path: dcm文件路径
    :return new_slices: 排序后的dcm文件列表
    """
    dcm_list = os.listdir(case_path)
    inumbers = []

    for slice in dcm_list:
        info = pdc.read_file(case_path + '/' + slice)
        inumbers.append(int(info.InstanceNumber))

    inarray = np.int_(inumbers)
    indices = inarray.argsort()
    new_slices = [dcm_list[indices[i]] for i in range(indices.shape[0])]

    return new_slices


def get_slice_thickness(path, dcm_list):
    """
    获取层厚信息
    :param path: 影像文件路径
    :param dcm_list: 排好序的DCM序列
    :return slice_thickness: float 层厚
    """
    pdc_slice0 = pdc.read_file(os.path.join(path, dcm_list[0]))
    slice_thickness = None
    try:
        slice_thickness = pdc_slice0.SliceThickness
    except AttributeError:
        pdc_slice1 = pdc.read_file(os.path.join(path, dcm_list[1]))
        try:
            slice_thickness = np.abs(pdc_slice0.SliceLocation - pdc_slice1.SliceLocation)
        except Exception:
            slice_thickness = np.abs(pdc_slice0.ImagePositionPatient[2] - pdc_slice0.ImagePositionPatient[2])

    return slice_thickness


def read_dcms(path):
    """
    读取dcm序列影像数据及体素间距
    :param path: dcm文件路径
    :return volumns: 三维影像数据
    :return spacing: numpy ndarray 体素间距, zyx
    """
    dcm_list = sort_dcms(path)    # 影像层断排序
    slice_thickness = get_slice_thickness(path, dcm_list)   # 利用pydicom读取体素间距

    volumns = np.zeros((len(dcm_list), 512, 512), dtype=np.float32)   # 初始化影像数组
    for i in range(len(dcm_list)):                                    # 按序读取影像
        dcm = dcm_list[i]
        slice = sitk.ReadImage(os.path.join(path, dcm))
        img = sitk.GetArrayFromImage(slice)
        img = np.squeeze(img)
        volumns[i, ...] = img

    # SimpleITK读取体素间距的层厚可能出错，需要按照pydicom处理
    dcm_tags = sitk.ReadImage(os.path.join(path, dcm_list[0]))
    x_pixel_spacing, y_pixel_spacing, thickness = dcm_tags.GetSpacing()
    if slice_thickness != thickness and slice_thickness is not None:
        thickness = slice_thickness
    spacing = np.array([thickness, y_pixel_spacing, x_pixel_spacing])

    return volumns, spacing