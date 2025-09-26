import torch
from torch.utils.data import Dataset
from data import read_img
import os
import numpy as np

class RibDataset(Dataset):
    '''
        预训练时的Dataset
    '''

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_names = os.listdir(data_dir)
        self.cube_paths = self.get_cube_paths()

    def get_cube_paths(self):
        cube_paths = []
        for folder_name in self.image_names:
            folder_path = os.path.join(self.data_dir, folder_name)
            cube_names = os.listdir(folder_path)
            for cube_name in cube_names:
                cube_path = os.path.join(folder_path, cube_name)
                cube_paths.append(cube_path)
        return cube_paths

    def __len__(self):
        return len(self.cube_paths)

    def __getitem__(self, idx):
        cube_path = self.cube_paths[idx]
        cube1_path = os.path.join(cube_path, "image.nii.gz")
        cube2_path = os.path.join(cube_path, "image_shift.nii.gz")

        cube1 = torch.from_numpy(read_img(cube1_path))
        cube2 = torch.from_numpy(read_img(cube2_path))

        cube1 = torch.unsqueeze(cube1, dim=0)
        cube2 = torch.unsqueeze(cube2, dim=0)

        return cube1, cube2


class RibDatasetWithMask(Dataset):
    '''
        训练时的Dataset
    '''
    def __init__(self, data_dir, train):
        self.data_dir = data_dir
        if train:
          self.file_names = ['FTRIB001_IMG.nii.gz', 'FTRIB003_IMG.nii.gz', 'FTRIB005_IMG.nii.gz', 'FTRIB009_IMG.nii.gz', 'FTRIB013_IMG.nii.gz', 'FTRIB014_IMG.nii.gz', 'FTRIB017_IMG.nii.gz', 'FTRIB022_IMG.nii.gz', 'FTRIB023_IMG.nii.gz', 'FTRIB024_IMG.nii.gz', 'FTRIB031_IMG.nii.gz', 'FTRIB034_IMG.nii.gz', 'FTRIB039_IMG.nii.gz', 'FTRIB041_IMG.nii.gz', 'FTRIB044_IMG.nii.gz', 'FTRIB047_IMG.nii.gz', 'FTRIB051_IMG.nii.gz', 'FTRIB055_IMG.nii.gz', 'FTRIB056_IMG.nii.gz', 'FTRIB057_IMG.nii.gz', 'FTRIB058_IMG.nii.gz', 'FTRIB061_IMG.nii.gz', 'FTRIB067_IMG.nii.gz', 'FTRIB077_IMG.nii.gz', 'FTRIB079_IMG.nii.gz', 'FTRIB082_IMG.nii.gz', 'FTRIB085_IMG.nii.gz', 'FTRIB087_IMG.nii.gz', 'FTRIB089_IMG.nii.gz', 'FTRIB091_IMG.nii.gz', 'FTRIB093_IMG.nii.gz', 'FTRIB098_IMG.nii.gz', 'FTRIB100_IMG.nii.gz', 'FTRIB101_IMG.nii.gz', 'FTRIB102_IMG.nii.gz', 'FTRIB108_IMG.nii.gz', 'FTRIB111_IMG.nii.gz', 'FTRIB113_IMG.nii.gz', 'FTRIB116_IMG.nii.gz', 'FTRIB118_IMG.nii.gz', 'FTRIB119_IMG.nii.gz', 'FTRIB123_IMG.nii.gz', 'FTRIB129_IMG.nii.gz', 'FTRIB131_IMG.nii.gz', 'FTRIB136_IMG.nii.gz', 'FTRIB141_IMG.nii.gz', 'FTRIB154_IMG.nii.gz', 'FTRIB155_IMG.nii.gz', 'FTRIB157_IMG.nii.gz', 'FTRIB161_IMG.nii.gz', 'FTRIB167_IMG.nii.gz', 'FTRIB171_IMG.nii.gz', 'FTRIB175_IMG.nii.gz', 'FTRIB178_IMG.nii.gz', 'FTRIB179_IMG.nii.gz', 'FTRIB180_IMG.nii.gz', 'FTRIB188_IMG.nii.gz', 'FTRIB189_IMG.nii.gz', 'FTRIB200_IMG.nii.gz', 'FTRIB202_IMG.nii.gz', 'FTRIB207_IMG.nii.gz', 'FTRIB209_IMG.nii.gz', 'FTRIB213_IMG.nii.gz', 'FTRIB214_IMG.nii.gz', 'FTRIB216_IMG.nii.gz', 'FTRIB219_IMG.nii.gz', 'FTRIB220_IMG.nii.gz', 'FTRIB221_IMG.nii.gz', 'FTRIB222_IMG.nii.gz', 'FTRIB223_IMG.nii.gz', 'FTRIB224_IMG.nii.gz', 'FTRIB229_IMG.nii.gz', 'FTRIB230_IMG.nii.gz', 'FTRIB234_IMG.nii.gz', 'FTRIB240_IMG.nii.gz', 'FTRIB245_IMG.nii.gz', 'FTRIB247_IMG.nii.gz', 'FTRIB258_IMG.nii.gz', 'FTRIB259_IMG.nii.gz', 'FTRIB261_IMG.nii.gz', 'FTRIB267_IMG.nii.gz', 'FTRIB268_IMG.nii.gz', 'FTRIB274_IMG.nii.gz', 'FTRIB277_IMG.nii.gz', 'FTRIB279_IMG.nii.gz', 'FTRIB283_IMG.nii.gz', 'FTRIB284_IMG.nii.gz', 'FTRIB287_IMG.nii.gz', 'FTRIB290_IMG.nii.gz', 'FTRIB291_IMG.nii.gz', 'FTRIB293_IMG.nii.gz', 'FTRIB304_IMG.nii.gz', 'FTRIB308_IMG.nii.gz', 'FTRIB310_IMG.nii.gz', 'FTRIB312_IMG.nii.gz', 'FTRIB314_IMG.nii.gz', 'FTRIB318_IMG.nii.gz', 'FTRIB326_IMG.nii.gz', 'FTRIB331_IMG.nii.gz', 'FTRIB337_IMG.nii.gz', 'FTRIB339_IMG.nii.gz', 'FTRIB383_IMG.nii.gz', 'FTRIB386_IMG.nii.gz', 'FTRIB387_IMG.nii.gz', 'FTRIB388_IMG.nii.gz', 'FTRIB390_IMG.nii.gz', 'FTRIB395_IMG.nii.gz', 'FTRIB396_IMG.nii.gz', 'FTRIB397_IMG.nii.gz', 'FTRIB403_IMG.nii.gz', 'FTRIB405_IMG.nii.gz', 'FTRIB409_IMG.nii.gz', 'FTRIB418_IMG.nii.gz', 'FTRIB451_IMG.nii.gz', 'FTRIB454_IMG.nii.gz', 'FTRIB456_IMG.nii.gz', 'FTRIB458_IMG.nii.gz', 'FTRIB462_IMG.nii.gz', 'FTRIB464_IMG.nii.gz', 'FTRIB472_IMG.nii.gz']
        else:
          self.file_names = os.listdir(data_dir)
        self.cube_paths = self.get_cube_paths()

    def get_cube_paths(self):
        cube_paths = []
        for folder_name in self.file_names:
            folder_path = os.path.join(self.data_dir, folder_name)
            cube_names = os.listdir(folder_path)
            for cube_name in cube_names:
                cube_path = os.path.join(folder_path, cube_name)
                cube_paths.append(cube_path)
        return cube_paths

    def __len__(self):
        return len(self.cube_paths)

    def data_process(self,label):
        label = (label > 0).to(torch.float)
        return label

    def __getitem__(self, idx):
        cube_path = self.cube_paths[idx]
        img_path = os.path.join(cube_path, 'img.npz')
        label_path = os.path.join(cube_path, 'msk.npz')
        img = torch.from_numpy(np.load(img_path)['arr_0'].astype(np.float32))
        label = torch.from_numpy(np.load(label_path)['arr_0'].astype(np.float32))
        label = self.data_process(label)

        img = torch.unsqueeze(img, 0)
        return img, label


class RibDatasetFinetune(Dataset):
    def __init__(self, data_dir, selected_file):
        self.data_dir = data_dir
        self.selected_file = selected_file
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
        return cube_paths

    def __len__(self):
        return len(self.cube_paths)

    def __getitem__(self, idx):
        cube_path = self.cube_paths[idx]
        img_path = os.path.join(cube_path, 'img.npz')
        label_path = os.path.join(cube_path, 'msk.npz')
        img = torch.from_numpy(np.load(img_path)['arr_0'].astype(np.float64))
        label = torch.from_numpy(np.load(label_path)['arr_0'].astype(np.float64))

        img = torch.unsqueeze(img, 0)
        return img, label


class RibDatasetAdaption(Dataset):
    def __init__(self, high_entropy_cube_dir, confident_cube_dir):
        self.high_entropy_cube_dir = high_entropy_cube_dir
        self.confident_cube_dir = confident_cube_dir
        self.cube_paths = self.get_cube_paths()

    def get_cube_paths(self):
        cube_paths = []
        file_names = os.listdir(self.high_entropy_cube_dir)
        for folder_name in file_names:
            folder_name = folder_name.split('\n')[0]
            folder_path = os.path.join(self.high_entropy_cube_dir, folder_name)
            cube_names = os.listdir(folder_path)
            for cube_name in cube_names:
                cube_path = os.path.join(folder_path, cube_name)
                cube_paths.append(cube_path)
        file_names = os.listdir(self.confident_cube_dir)
        for folder_name in file_names:
            folder_name = folder_name.split('\n')[0]
            folder_path = os.path.join(self.confident_cube_dir, folder_name)
            cube_names = os.listdir(folder_path)
            for cube_name in cube_names:
                cube_path = os.path.join(folder_path, cube_name)
                cube_paths.append(cube_path)
        return cube_paths

    def data_process(self,label):
        label = (label > 0).to(torch.float)
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
