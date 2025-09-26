import argparse
import os
import shutil

from models.moco.builder import MoCoV3
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models.CNNEncoder
# import models.UNet_3DEncoder
import models.FracNet.encoder
# import models.encoder
import torch.nn.functional as F
from tqdm import tqdm

from data.ribDataset import RibDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed

parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--data', default='/data/yyh/Processed Data/ribSegV2/ribSegv2_pretrain')
#################################################################################################
parser.add_argument('--valid-data', default='/data/yyh/Processed Data/cube/pretrain_val')
parser.add_argument('--gpu', default='0', type=int)
parser.add_argument('--save-dir', default='checkpoints/ribSegV2/FracNet/pretrain', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-3, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--moco-t', default=0.3, type=float)
parser.add_argument('--arch', default='cnn')
parser.add_argument('--resume', default=True, type=bool)
parser.add_argument('--resume-epoch', default=16, type=int)



def main():
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1, 2'

    # model
    model = MoCoV3(
        base_encoder=models.FracNet.encoder.Encoder(in_channels=1)
    )

    if args.resume:
        # 获取参数的状态字典
        weight_path = "checkpoints/ribSegV2/FracNet/pretrain/checkpoint_epoch15.pth.tar"
        pretrain_weight = torch.load(weight_path, map_location='cuda:{}'.format(args.gpu))
        state_dict = pretrain_weight['state_dict']

        model_dict = model.state_dict()
        for name, param in state_dict.items():
            # print(name)
            if name in model_dict.keys():
                model_dict[name] = param
            else:
                print('{}不在模型中'.format(name))
        model.load_state_dict(model_dict)
        print('模型{}已成功加载，现在继续训练'.format(weight_path))

    # dist cuda
    model = model.cuda(args.gpu)
    print("Total number of param in model is ", sum(x.numel() for x in model.parameters()))

    # Loss用的是自己写的PixelContrastive，optimizer用SGD
    criterion = ContrastiveLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # data
    pretrain_dir = args.data
    rib_dataset = RibDataset(pretrain_dir)
    rib_dataloader = DataLoader(rib_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, num_workers=2, pin_memory=True)

    # train
    train_start = time.time()
    tb_writer = SummaryWriter(log_dir='./log/ribSegV2/FracNet/pretrain')

    begin_epoch = args.resume_epoch if args.resume else 0
    for epoch in range(begin_epoch, args.epochs):
        # train for one epoch
        train(rib_dataloader, model, criterion, optimizer, epoch, args, tb_writer)


    train_end = time.time()

    print('total training time elapses {} hours'.format((train_end - train_start) / 3600.0))



def train(dataloader, model, criterion, optimizer, epoch, args, tb_writer):
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()


    print('epoch{} start'.format(epoch), '*' * 16)
    # 将losses重置，每个epoch输出一次平均losses
    losses.reset()

    for cube1, cube2 in tqdm(dataloader):

        img_q = cube1.cuda(args.gpu)
        img_k = cube2.cuda(args.gpu)
        q, k = model(x1=img_q, x2=img_k)
        loss = criterion(q, k, args.moco_t)
        loss.requires_grad_(True)
        print('epoch{}_loss:{}'.format(epoch, loss))
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(losses)
    tb_writer.add_scalar("PreTrain/Loss", losses.avg, epoch)
    # measure elapsed time
    save_checkpoint({
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, root=args.save_dir, filename='checkpoint_epoch{}.pth.tar'.format(epoch))


# 定义上采样模块
class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.up = Upsample(scale_factor=16)


    def cal_loss(self, f1, f2, dis_thre, t):
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        # 1.计算f1到f2的欧氏距离，根据距离阈值确定正负样本
        euclidean_distance = torch.sqrt(torch.sum((f1 - f2) ** 2, dim=-1))

        # print(torch.unique(euclidean_distance, return_counts=True))
        label = (euclidean_distance <= dis_thre)
        # print(torch.unique(label, return_counts=True))
        # print('label:', label.shape)

        # 2.计算f1到f2的余弦相似度
        dot_product = torch.sum(f1 * f2, dim=-1)
        norm1 = torch.norm(f1, dim=-1)
        norm2 = torch.norm(f2, dim=-1)
        sim = dot_product / (norm1 * norm2)
        sim = sim / t
        # print('sim:', sim.shape)

        # 3.计算像素对比损失
        pos_sim = sim.masked_fill(~label, 0)
        pos_sim = torch.sum(torch.exp(pos_sim), dim=-1)
        neg_sim = sim.masked_fill(label, 0)
        neg_sim = torch.sum(torch.exp(neg_sim), dim=-1)
        loss_pix = - torch.log(pos_sim / (pos_sim + neg_sim))
        loss_pix = loss_pix.mean()

        return loss_pix

    def forward(self, f1, f2, t):
        # 将特征上采样回原来的尺寸
        f1 = self.up(f1)
        f2 = self.up(f2)

        loss_pix1 = self.cal_loss(f1, f2, dis_thre=0.7, t=t)
        loss_pix2 = self.cal_loss(f2, f1, dis_thre=0.7, t=t)
        loss = (loss_pix1 + loss_pix2) / 2

        return loss


def save_checkpoint(state, is_best, root, filename='checkpoints.pth.tar'):
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'model_best.pth.tar'))


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