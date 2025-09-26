import argparse
import csv

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models.res_unet
import models.unet
import models.atten_unet.atten_unet_finetune
import models.FracNet.fracNet
import SimpleITK as sitk
from data import test_process, crop_cube, post_process
from torch.nn.parallel import DataParallel
from skimage import measure
from skimage.morphology import label
# #程序中链接了多个 OpenMP 运行时库的副本
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--test_img', default=r'/me4012/RibSegV2/data/ribfrac-test-images/')
parser.add_argument('--test_msk', default=r'/me4012/RibSegV2/data/label/seg/')
parser.add_argument('--gpu', default='0', type=int)
peth=0.6
test_type = "ALL_36000"
test_epoch=99
def main():

    args = parser.parse_args()

    # model
    model = models.atten_unet.atten_unet_finetune.UNet3D(in_channels=1, out_channels=1)
    model.cuda()  

    weight_path = "/me4012/RibSegV2/checkpoints/AblationStudy/no{}/checkpoint_epoch{}.pth.tar".format(test_type, test_epoch)

    pretrain_weight = torch.load(weight_path, map_location='cuda:0')
    state_dict = pretrain_weight['state_dict']
    # state_dict = pretrain_weight['net']

    model_dict = model.state_dict()
    for name, param in state_dict.items():
        # name = name[7:]
        # print(name)
        # if 'encoder.' in name:
        #     name = name.split('encoder.')[1]
        if name in model_dict.keys():
            model_dict[name] = param
        else:
            print('{}不在模型中'.format(name))
    model.load_state_dict(model_dict)
    print('{}模型加载完毕'.format(weight_path))

    cudnn.benchmark = True

    predict(model, args)


@torch.no_grad()
def predict(model, args):

    # switch to evaluate mode
    model.eval()

    IOU = AverageMeter('test_iou')
    DICE = AverageMeter('test_dice')

    IOU.reset()
    DICE.reset()
    cubes = test_process(args.test_img, args.test_msk, pre_handle=True)
    index = 0
    for cube, shape, img_name, mask in cubes:
        prediction = torch.zeros(shape).cuda()
        for img, position in cube:
            zmin, ymin, xmin, zmax, ymax, xmax = position
            # get the inputs and wrap in Variable
            img = torch.from_numpy(img).cuda().to(torch.float32)

            # forward
            feature = model(x=img)
            cube_pre = (feature > 0).squeeze()
            if xmax - xmin == 64 and ymax - ymin == 64 and zmax - zmin == 64:
                prediction[zmin:zmax, ymin:ymax, xmin:xmax] = prediction[zmin:zmax, ymin:ymax, xmin:xmax] + cube_pre
            else:
                prediction[zmin:zmax, ymin:ymax, xmin:xmax] = prediction[zmin:zmax, ymin:ymax, xmin:xmax] + cube_pre[:(zmax-zmin), :(ymax-ymin), :(xmax-xmin)]
        prediction = prediction.to(torch.int8).cpu()
        print(torch.unique(prediction, return_counts=True))
        prediction = (prediction > 3).to(torch.int8)
        prediction = post_process(prediction)
        iou, dice = metric(prediction, mask)
        print('index:{} iou:{} dice:{}'.format(index, iou, dice))
        pre = sitk.GetImageFromArray(prediction)
        sitk.WriteImage(pre, "/me4012/RibSegV2/data/yyh_2025/PEth0901/prediction_{}/no{}/".format(test_epoch, test_type) + img_name + '/prediction.nii.gz')

        IOU.update(iou)
        DICE.update(dice)
        index += 1
    print('IOU:{} DICE:{}'.format(IOU, DICE))


def metric(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = intersection.float() / union.float()
    dice = (2 * intersection.float()) / (pred_mask.sum() + true_mask.sum())
    return iou, dice


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()