import os
import random
import torch
import torch.nn.functional as F
import random
import torch.utils.data as data
from scipy.ndimage.interpolation import zoom

import numpy as np


class RibDataset(data.Dataset):
    def __init__(self, flist_file, stage="train"):
        with open(flist_file, "r") as file:
            all_list = file.readlines()
        if stage == "train":
            self.img_list = self.choice_list(all_list, ratio=0.5)
        else:
            self.img_list = self.choice_list(all_list, ratio=0.5)

    @staticmethod
    def choice_list(all_list, ratio=0.5):
        sample_num = int(len(all_list) * ratio)
        img_list = random.sample(all_list, sample_num)

        return img_list

    def __getitem__(self, idx):
        arrs = np.load(self.img_list[idx].rstrip())
        img = arrs["img"].astype(np.float32)
        msk = arrs["msk"].astype(np.float32)
        # img = self.process(img)
        # msk = self.process(msk)   # zoom插值后，会出现很多小于255的值，会破坏msk二值化
        # msk[msk >= 130] = 255    # 将大于220的值设为255，小于220的值保持不变，适应肺结节边界难以确定的情况
        # msk[msk < 130] = 0
        img, msk = self.random_flip(img, msk)
        img, msk = self.random_rotate(img, msk)
        img = np.expand_dims(img, 0)
        msk = np.expand_dims(msk, 0)
        img /= 255.
        # msk /= 255.

        return img.copy(), msk.copy()

    def process(self, img):
        depth, height, width = img.shape
        zoom_d, zoom_h, zoom_w = self.size[0] / depth, self.size[1] / height, self.size[2] / width
        zoomed = zoom(img, [zoom_d, zoom_h, zoom_w])

        return zoomed

    def process_gpu(self, img):
        new_shape = np.array(list(self.size), dtype=np.float32)
        old_shape = np.array(list(img.shape), dtype=np.float32)
        real_resize_factor = new_shape / old_shape
        image = np.expand_dims(img, axis=0)
        image = np.expand_dims(image, axis=1)
        input_tsr = torch.from_numpy(image).cuda()
        resized_tsr = F.interpolate(input_tsr, scale_factor=tuple(real_resize_factor), mode='nearest')
        image = resized_tsr.squeeze().cpu().data.numpy()

        return image

    @staticmethod
    def random_flip(img, msk):
        axis = random.randint(-1, 2)
        if axis == 0:
            r_img = img[::-1, :, :]
            r_msk = msk[::-1, :, :]
        elif axis == 1:
            r_img = img[:, ::-1, :]
            r_msk = msk[:, ::-1, :]
        elif axis == 2:
            r_img = img[:, :, ::-1]
            r_msk = msk[:, :, ::-1]
        else:
            return img.copy(), msk.copy()

        return r_img, r_msk

    @staticmethod
    def random_rotate(img, msk):
        angle = random.choice([1, 2, 3])
        axis = random.choice([0, -1])
        # axis = random.choice([0, 1, 2, -1])
        img_rotated = np.empty(img.shape)
        msk_rotated = np.empty(msk.shape)
        if axis == -1:
            return img.copy(), msk.copy()
        elif axis == 0:
            for z in range(img.shape[0]):
                img_slice = img[z, ...]
                msk_slice = msk[z, ...]
                img_rotated[z, ...] = np.rot90(img_slice, angle)
                msk_rotated[z, ...] = np.rot90(msk_slice, angle)
        elif axis == 1:
            for y in range(img.shape[1]):
                img_slice = img[:, y, :]
                msk_slice = msk[:, y, :]
                img_rotated[:, y, :] = np.rot90(img_slice, angle)
                msk_rotated[:, y, :] = np.rot90(msk_slice, angle)
        elif axis == 2:
            for x in range(img.shape[2]):
                img_slice = img[..., x]
                msk_slice = msk[..., x]
                img_rotated[..., x] = np.rot90(img_slice, angle)
                msk_rotated[..., x] = np.rot90(msk_slice, angle)

        return img_rotated, msk_rotated

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    img_file = "../data/cube32_train.csv"

    dataset = RibDataset(img_file)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=False)

    for batch_idx, (img, msk) in enumerate(loader):
        print(img.size(), msk.size())
