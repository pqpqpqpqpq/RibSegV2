import torch
from torch.utils.data import Dataset
from data import read_img
import SimpleITK as sitk
import os
import numpy as np


class RibSegV2(Dataset):
    def __init__(self, data_dir, train):
        self.data_dir = data_dir
        if not train:
            self.file_names = os.listdir(data_dir)
        else:
            self.random = ['RibFrac403-image.nii.gz', 'RibFrac139-image.nii.gz', 'RibFrac365-image.nii.gz', 'RibFrac345-image.nii.gz', 'RibFrac16-image.nii.gz', 'RibFrac159-image.nii.gz', 'RibFrac287-image.nii.gz', 'RibFrac134-image.nii.gz', 'RibFrac237-image.nii.gz', 'RibFrac82-image.nii.gz', 'RibFrac162-image.nii.gz', 'RibFrac101-image.nii.gz', 'RibFrac241-image.nii.gz', 'RibFrac207-image.nii.gz', 'RibFrac411-image.nii.gz', 'RibFrac120-image.nii.gz', 'RibFrac99-image.nii.gz', 'RibFrac346-image.nii.gz', 'RibFrac245-image.nii.gz', 'RibFrac98-image.nii.gz', ]
            self.selected_1 = ['RibFrac32-image.nii.gz', 'RibFrac381-image.nii.gz', 'RibFrac124-image.nii.gz', 'RibFrac37-image.nii.gz', 'RibFrac200-image.nii.gz', 'RibFrac353-image.nii.gz', 'RibFrac109-image.nii.gz', 'RibFrac232-image.nii.gz', 'RibFrac348-image.nii.gz', 'RibFrac163-image.nii.gz', 'RibFrac378-image.nii.gz', 'RibFrac91-image.nii.gz', 'RibFrac63-image.nii.gz', 'RibFrac310-image.nii.gz', 'RibFrac283-image.nii.gz', 'RibFrac65-image.nii.gz', 'RibFrac241-image.nii.gz', 'RibFrac40-image.nii.gz', 'RibFrac187-image.nii.gz', 'RibFrac355-image.nii.gz']
            self.selected_2 = ['RibFrac172-image.nii.gz', 'RibFrac39-image.nii.gz', 'RibFrac372-image.nii.gz', 'RibFrac149-image.nii.gz', 'RibFrac136-image.nii.gz', 'RibFrac111-image.nii.gz', 'RibFrac133-image.nii.gz', 'RibFrac157-image.nii.gz', 'RibFrac384-image.nii.gz', 'RibFrac319-image.nii.gz', 'RibFrac139-image.nii.gz', 'RibFrac284-image.nii.gz', 'RibFrac184-image.nii.gz', 'RibFrac322-image.nii.gz', 'RibFrac164-image.nii.gz', 'RibFrac106-image.nii.gz', 'RibFrac331-image.nii.gz', 'RibFrac246-image.nii.gz', 'RibFrac287-image.nii.gz', 'RibFrac290-image.nii.gz']
            self.selected_3 = ['RibFrac273-image.nii.gz', 'RibFrac420-image.nii.gz', 'RibFrac383-image.nii.gz', 'RibFrac97-image.nii.gz', 'RibFrac148-image.nii.gz', 'RibFrac135-image.nii.gz', 'RibFrac104-image.nii.gz', 'RibFrac78-image.nii.gz', 'RibFrac289-image.nii.gz', 'RibFrac19-image.nii.gz', 'RibFrac76-image.nii.gz', 'RibFrac120-image.nii.gz', 'RibFrac38-image.nii.gz', 'RibFrac57-image.nii.gz', 'RibFrac267-image.nii.gz', 'RibFrac399-image.nii.gz', 'RibFrac243-image.nii.gz', 'RibFrac62-image.nii.gz', 'RibFrac388-image.nii.gz', 'RibFrac81-image.nii.gz']
            self.selected_4 = ['RibFrac132-image.nii.gz', 'RibFrac33-image.nii.gz', 'RibFrac137-image.nii.gz', 'RibFrac343-image.nii.gz', 'RibFrac249-image.nii.gz', 'RibFrac230-image.nii.gz', 'RibFrac143-image.nii.gz', 'RibFrac117-image.nii.gz', 'RibFrac351-image.nii.gz', 'RibFrac237-image.nii.gz', 'RibFrac58-image.nii.gz', 'RibFrac96-image.nii.gz', 'RibFrac151-image.nii.gz', 'RibFrac20-image.nii.gz', 'RibFrac118-image.nii.gz', 'RibFrac101-image.nii.gz', 'RibFrac99-image.nii.gz', 'RibFrac279-image.nii.gz', 'RibFrac162-image.nii.gz', 'RibFrac50-image.nii.gz']
            self.selected_5 = ['RibFrac134-image.nii.gz', 'RibFrac94-image.nii.gz', 'RibFrac179-image.nii.gz', 'RibFrac177-image.nii.gz', 'RibFrac313-image.nii.gz', 'RibFrac195-image.nii.gz', 'RibFrac316-image.nii.gz', 'RibFrac349-image.nii.gz', 'RibFrac113-image.nii.gz', 'RibFrac210-image.nii.gz', 'RibFrac332-image.nii.gz', 'RibFrac397-image.nii.gz', 'RibFrac75-image.nii.gz', 'RibFrac171-image.nii.gz', 'RibFrac410-image.nii.gz', 'RibFrac47-image.nii.gz', 'RibFrac359-image.nii.gz', 'RibFrac345-image.nii.gz', 'RibFrac253-image.nii.gz', 'RibFrac159-image.nii.gz']
            self.confident = ['RibFrac112-image.nii.gz', 'RibFrac131-image.nii.gz', 'RibFrac165-image.nii.gz', 'RibFrac220-image.nii.gz', 'RibFrac225-image.nii.gz', 'RibFrac226-image.nii.gz', 'RibFrac26-image.nii.gz', 'RibFrac27-image.nii.gz', 'RibFrac312-image.nii.gz', 'RibFrac367-image.nii.gz', 'RibFrac411-image.nii.gz', 'RibFrac418-image.nii.gz', 'RibFrac5-image.nii.gz']
            self.file_names = set(self.random+self.selected_1+self.selected_2+self.selected_3+self.selected_4+self.selected_5+self.confident)
        self.cube_paths = self.get_cube_paths()

    def get_cube_paths(self):
        cube_paths = []
        for folder_name in self.file_names:
            folder_name = folder_name.split('-image.nii.gz')[0]
            folder_path = os.path.join(self.data_dir, folder_name)
            cube_names = os.listdir(folder_path)
            for cube_name in cube_names:
                cube_path = os.path.join(folder_path, cube_name)
                cube_paths.append(cube_path)
        return cube_paths

    @staticmethod
    def data_process(label):
        label = (label > 0).to(torch.int)
        return label

    def __len__(self):
        return len(self.cube_paths)

    def __getitem__(self, idx):
        cube_path = self.cube_paths[idx]
        img_path = os.path.join(cube_path, 'image.nii.gz')
        label_path = os.path.join(cube_path, 'mask.nii.gz')
        img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32))
        label = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32))
        label = self.data_process(label)
        img = torch.unsqueeze(img, 0)
        return img, label


class RibSegV2Finetune(Dataset):
    def __init__(self, data_dir, selected_file, pra, self_train):
        self.data_dir = data_dir
        self.selected_file = selected_file
        self.pra = pra
        self.self_train = self_train
        self.cube_paths = self.get_cube_paths()

    def get_cube_paths(self):
        cube_paths = []
        file = open(self.selected_file)
        file_names = file.readlines()
        for folder_name in file_names:
            folder_name = folder_name.split('\n')[0]
            folder_path = os.path.join(self.data_dir, folder_name)
            cube_names = os.listdir(folder_path)
            for cube_name in cube_names:
                cube_path = os.path.join(folder_path, cube_name)
                cube_paths.append(cube_path)
        if self.pra:
            high_entropy_path = '/data/yyh/Processed Data/high_entropy_cube/ribsegV2/FracNet'
            files = os.listdir(high_entropy_path)
            for file in files:
                cubes = os.listdir(os.path.join(high_entropy_path, file))
                for cube in cubes:
                    cube_path = os.path.join(high_entropy_path, file, cube)
                    cube_paths.append(cube_path)
        if self.self_train:
            confident_cube_path = '/data/yyh/Processed Data/confident_cube/FracNet'
            files = os.listdir(confident_cube_path)
            for file in files:
                cubes = os.listdir(os.path.join(confident_cube_path, file))
                for cube in cubes:
                    cube_path = os.path.join(confident_cube_path, file, cube)
                    cube_paths.append(cube_path)
        return cube_paths

    @staticmethod
    def data_process(label):
        label = (label > 0).to(torch.float32)
        return label

    def __len__(self):
        return len(self.cube_paths)

    def __getitem__(self, idx):
        cube_path = self.cube_paths[idx]
        img_path = os.path.join(cube_path, 'img.npz')
        label_path = os.path.join(cube_path, 'msk.npz')
        img = torch.from_numpy(np.load(img_path)['arr_0'].astype(np.float32))
        label = torch.from_numpy(np.load(label_path)['arr_0'].astype(np.float32))
        label = self.data_process(label)
        img = torch.unsqueeze(img, 0)
        return img, label
