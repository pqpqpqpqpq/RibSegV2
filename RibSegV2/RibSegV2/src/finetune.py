import argparse
import os
import shutil
import time
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models.unet
import models.res_unet
import models.atten_unet.atten_unet_finetune
import models.FracNet.fracNet
# from dataset.ribDataset import RibDatasetWithMask, RibDatasetFinetune
from data.ribSegDataset import RibSegV2, RibSegV2Finetune
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tools.metric import metric
from querySample import QuerySample


parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--train_data', default='/me4012/RibSegV2/data/Processed Data/ribseg_v2/train/')
parser.add_argument('--val_data', default='/me4012/RibSegV2/data/Processed Data/ribseg_v2/val/')
parser.add_argument('--gpu', default='0', type=int)
parser.add_argument('--save-dir', default='checkpoints/ribSegV2/CL+AL+ST/', type=str)
# parser.add_argument('--save-dir', default='checkpoints/attUnet/dbal_finetune/decoder_240_withoutLow_freezeEncoder', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 300], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--moco-t', default=0.07, type=float)
parser.add_argument('--arch', default='cnn')
parser.add_argument('--finetune', default=True, type=bool)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume-epoch', default=273, type=int)
parser.add_argument('--whole-data', default=True, type=bool)
parser.add_argument('--pra', default=False, type=bool)
parser.add_argument('--self-train', default=False, type=bool)
parser.add_argument('--valid-gpu', default=0, type=int)

def main():
    args = parser.parse_args()
    begin_epoch = 0

    # model
    model = models.atten_unet.atten_unet_finetune.UNet3D(in_channels=1, out_channels=1)
    # model = models.FracNet.fracNet.FracNetFinetune(in_channels=1, num_classes=1)

    if args.finetune:
        # 获取参数的状态字典
        weight_path = "checkpoints/ribSegV2/pretrain/checkpoint_epoch80.pth.tar"
        pretrain_weight = torch.load(weight_path, map_location='cuda:{}'.format(args.valid_gpu))
        state_dict = pretrain_weight['state_dict']

        model_dict = model.state_dict()
        for name, param in state_dict.items():
            # name = name.split('module.')[1]
            # 提取encoder_q的权重给model
            if 'encoder_q' in name:
                name = name.split('encoder_q.')[0] + name.split('encoder_q.')[1]
                if name in model_dict.keys():
                    # print(name)
                    model_dict[name] = param
                    continue
            # 提取投影头的权重给model
            if 'projector' in name:
                if name in model_dict.keys():
                    model_dict[name] = param
                    continue
                else:
                    print('{}不在模型中'.format(name))
            else:
                print('{}不在模型中'.format(name))
        model.load_state_dict(model_dict)
        print('模型{}已加载完毕'.format(weight_path))


    if args.resume:
        begin_epoch = args.resume_epoch
        # model = DataParallel(model)
        # 获取参数的状态字典
        weight_path = "checkpoints/ribSegV2/FracNet/active_rpa_self_train/checkpoint_epoch272.pth.tar"

        pretrain_weight = torch.load(weight_path, map_location='cuda:{}'.format(args.valid_gpu))
        state_dict = pretrain_weight['state_dict']

        model_dict = model.state_dict()
        # print(state_dict)
        # print(model_dict)
        for name, param in state_dict.items():
            # name1 = name.split('module.')[1]
            if name in model_dict.keys():
                model_dict[name] = param
            else:
                print('{}不在模型中'.format(name))
        model.load_state_dict(model_dict)
        print(weight_path+'模型加载完毕')

    model = model.cuda(args.valid_gpu)

    # Loss
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # data
    traindir = args.train_data
    valdir = args.val_data

    train_dataset = RibSegV2(traindir, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, sampler=None)
    val_dataset = RibSegV2(valdir, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=None)

    # train
    train_start = time.time()
    tb_writer1 = SummaryWriter(log_dir='./log/ribSegV2/CL+AL+ST/train')
    tb_writer2 = SummaryWriter(log_dir='./log/ribSegV2/CL+AL+ST/val')

    for epoch in range(begin_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        if not args.whole_data and epoch % 50 == 0 and epoch != 0:
            q = QuerySample(model, epoch, args)
            q.init_params_and_folder()
            q.query()
            train_dataset = RibSegV2Finetune(traindir, '/data/yyh/ribseg_v2/FracNet_selected_sample_0.txt', pra=True, self_train=args.self_train)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, sampler=None)
        train(train_dataloader, model, optimizer, epoch, args, tb_writer1)
        if epoch % 10 == 0:
            validation(val_dataloader, model, epoch, tb_writer2, args)

    train_end = time.time()

    print('total training time elapses {} hours'.format((train_end - train_start) / 3600.0))


def train(dataloader, model, optimizer, epoch, args, tb_writer1):
    losses = AverageMeter('Loss', ':.4e')
    ious = AverageMeter('Loss', ':.4e')
    dices = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()

    print('epoch{} start'.format(epoch), '*' * 16)
    # 将losses重置，每个epoch输出一次平均losses
    losses.reset()
    ious.reset()
    dices.reset()

    for img, label in tqdm(dataloader):
        # [8, 1, 64, 64, 64] B,C,D,H,W
        img = img.to(torch.float32).cuda(args.valid_gpu, non_blocking=True)
        label = label.to(torch.float32).cuda(args.valid_gpu, non_blocking=True)
        features = model(x=img)

        loss_arr = torch.zeros(len(features)).cuda(args.valid_gpu)
        iou_arr = torch.zeros(len(features)).cuda(args.valid_gpu)
        dice_arr = torch.zeros(len(features)).cuda(args.valid_gpu)
        for i in range(len(features)):
            loss_arr[i] = cal_loss(features[i], label[i])
            cube_pre = (features[i] > 0).squeeze().cuda(args.valid_gpu)
            iou_arr[i], dice_arr[i] = metric(cube_pre, label[i])
        loss = torch.mean(loss_arr)
        iou = torch.mean(iou_arr)
        dice = torch.mean(dice_arr)
        loss.requires_grad_(True)
        losses.update(loss, args.batch_size)
        ious.update(iou, args.batch_size)
        dices.update(dice, args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch{} train loss:{} iou:{} dice:{}'.format(epoch, loss, iou, dice))
    print('mean losses:', losses)
    print('mean ious:', ious)
    print('mean dice:', dices)

    tb_writer1.add_scalar("Loss", losses.avg, epoch)
    tb_writer1.add_scalar("IOU", ious.avg, epoch)
    tb_writer1.add_scalar("Dice", dices.avg, epoch)
    # measure elapsed time
    save_checkpoint({
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, root=args.save_dir, filename='checkpoint_epoch{}.pth.tar'.format(epoch))


@torch.no_grad()
def validation(dataloader, model, epoch, tb_writer2, args):
    losses = AverageMeter('Loss', ':.4e')
    ious = AverageMeter('Loss', ':.4e')
    dices = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.eval()

    print('epoch{} start'.format(epoch), '*' * 16)
    # 将losses重置，每个epoch输出一次平均losses
    losses.reset()
    ious.reset()
    dices.reset()

    for img, label in tqdm(dataloader):

        # [8, 1, 64, 64, 64] B,C,D,H,W
        img = img.cuda(args.valid_gpu, non_blocking=True).to(torch.float32)
        feature = model(x=img)
        label = label.cuda(args.valid_gpu, non_blocking=True).to(torch.float32)
        loss = cal_loss(feature, label)
        cube_pre = (feature > 0).squeeze()
        iou, dice = metric(cube_pre, label)

        losses.update(loss)
        ious.update(iou)
        dices.update(dice)
    print('losses:{},iou:{},dice:{}'.format(losses, ious, dices))

    tb_writer2.add_scalar("Loss", losses.avg, epoch)
    tb_writer2.add_scalar("IOU", ious.avg, epoch)
    tb_writer2.add_scalar("Dice", dices.avg, epoch)


def cal_loss(pre, target):
    pre = torch.squeeze(pre)
    target = torch.squeeze(target)
    bce = F.binary_cross_entropy_with_logits(pre, target)
    return bce


def save_checkpoint(state, is_best, root, filename='checkpoints.pth.tar'):
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()