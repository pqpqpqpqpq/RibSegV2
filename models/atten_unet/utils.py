import cv2

import numpy as np
import SimpleITK as sitk

from skimage import measure
from skimage.morphology import binary_closing, ball, disk
from scipy.ndimage import binary_closing


def largest_label_image(image, bg=-1):
    vals, counts = np.unique(image, return_counts=True)  # 统计image中的值及频率

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        maxind = np.argmax(counts)
        return vals[maxind], counts[maxind]
    else:
        return None, None


def segment_lung_mask_slice_fast(image, hu_threshold, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > hu_threshold, dtype=np.int8) + 1
    # get through the light line at the bottom
    binary_image = np.pad(binary_image, ((2, 0), (0, 0)), 'constant', constant_values=((2, 2), (2, 2)))
    binary_image[0, :] = 1
    binary_image[:, 0] = 1
    binary_image[:, -1] = 1
    # 获得阈值图像
    labels = measure.label(binary_image, connectivity=1)
    # label()函数标记连通区域
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0]
    # Fill the air around the person
    binary_image[labels == background_label] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    binary_image -= 1  # Make the image actual binary
    if fill_lung_structures:
        labeling = measure.label(binary_image)
        l_max, _ = largest_label_image(labeling, bg=labeling[0, 0])
        if l_max is not None:  # This slice contains some lung
            binary_image[np.logical_and(labeling != l_max, labeling != labeling[0, 0])] = 1
    binary_image = 1 - binary_image  # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max, maxcount = largest_label_image(labels, bg=0)
    if l_max is not None:  # There are air pockets
        background_label = labels[0, 0]
        min_label = labels.min()
        _, labelcounts = np.unique(labels, return_counts=True)
        maxlabel = labels.max()
        label_coords = np.where(labels != background_label)
        for lci in range(len(label_coords[0])):
            label = labels[label_coords[0][lci]][label_coords[1][lci]] - min_label
            if labelcounts[label] < maxcount / 4:
                binary_image[label_coords[0][lci]][label_coords[1][lci]] = 0

    if maxcount is None or maxcount < image.size * 0.01:
        return np.zeros_like(image, dtype=np.int8)
    binary_image = np.delete(binary_image, [0, 1], axis=0)
    return binary_image


def extend_mask(imagemask, ball_size=4, iterations=10):
    padding = ball_size * iterations
    mask = np.zeros(
        shape=(imagemask.shape[0] + 2 * padding, imagemask.shape[1] + 2 * padding, imagemask.shape[2] + 2 * padding),
        dtype=bool)
    mask[padding:imagemask.shape[0] + padding, padding:imagemask.shape[1] + padding,
    padding:imagemask.shape[2] + padding][imagemask == 1] = imagemask[imagemask == 1]

    struct = ball(ball_size)
    mask = binary_closing(mask, structure=struct, iterations=iterations)
    return mask[padding:imagemask.shape[0] + padding, padding:imagemask.shape[1] + padding,
           padding:imagemask.shape[2] + padding]


def del_surplus(mask, image):
    """
    提取边界框
    :param mask: mask
    :param image:  影像
    :return:
    """
    zs, ys, xs = np.where(mask)
    zmin, zmax = np.min(zs), np.max(zs)
    ymin, ymax = np.min(ys), np.max(ys)
    xmin, xmax = np.min(xs), np.max(xs)

    lung_mask_del = mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    image_del = image[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

    return lung_mask_del, image_del, (zmin, ymin, xmin, zmax, ymax, xmax)


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

