import argparse
import os
import shutil

from tqdm import tqdm

import math
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.res_unet import ResUNet
from tools.metric import MetricTracker, dice_coeff
from dataset.ribDataset import RibDatasetWithMask
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tools.logger import MyWriter

parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--train_data', default='E:\cube\\train')
parser.add_argument('--val_data', default='E:\cube\\val')
parser.add_argument('--gpu', default='0', type=int)
parser.add_argument('--save-dir', default='checkpoints/res_unet', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--moco-t', default=0.3, type=float)
parser.add_argument('--arch', default='cnn')


def main():
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1, 2, 3'


    # model
    model = ResUNet(in_channels=1, num_classes=1)
    # print(model)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # data
    train_dir = args.train_data
    val_dir = args.val_data
    train_dataset = RibDatasetWithMask(train_dir)
    val_dataset = RibDatasetWithMask(val_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=None)

    # writer
    writer = MyWriter("log/res_Unet")
    # train
    step = 0
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        print('lr:', args.lr)
        # train for one epoch
        train(train_dataloader, model, optimizer, epoch, args)
        if step % 5 == 0:
            valid_metrics = validation(val_dataloader, model, writer, step)
            save_path = os.path.join(
                args.save_dir, "{}_checkpoint_{}.pt".format('valid', step)
            )

            best_loss = min(valid_metrics["valid_loss"], best_loss)
            best_iou = max(valid_metrics["valid_iou"], best_iou)
            best_dice = max(valid_metrics["valid_dice"], best_dice)
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "arch": "ResUnet",
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            print("Saved checkpoint to: %s" % save_path)
        step += 1


def train(dataloader, model, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()
    model.cuda(args.gpu)
    # # 检查CUDA是否可用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)  # 使用DataParallel在多个GPU上运行模型

    tb_writer = SummaryWriter(log_dir='./log/res_unet')
    print('epoch{} start'.format(epoch), '*' * 128)
    # 将losses重置，每个epoch输出一次平均losses
    losses.reset()

    for idx, img, label, position in dataloader:

        # [8, 1, 64, 64, 64] B,C,D,H,W
        img = img.cuda(args.gpu, non_blocking=True).to(torch.float32)
        features = model(x=img).cpu()
        loss_arr = torch.zeros(len(features))
        for i in range(len(features)):
            loss_arr[i] = cal_loss(features[i], label[i])
        loss = torch.mean(loss_arr)
        print('epoch{} train loss:{}'.format(epoch, loss))
        loss.requires_grad_(True)
        losses.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('mean losses:', losses)


    tb_writer.add_scalar("Train/Loss", losses.avg, epoch)
    # measure elapsed time
    save_checkpoint({
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, root=args.save_dir, filename='checkpoint_epoch{}.pth.tar'.format(epoch))


def validation(valid_loader, model, logger, step):

    # logging accuracy and loss
    valid_acc = MetricTracker()
    valid_loss = MetricTracker()
    valid_iou = AverageMeter('IoU', ':.4e')
    valid_dice = AverageMeter('Dice', ':.4e')

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, img, label, position in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        img = img.cuda(non_blocking=True)

        # forward
        features = model(x=img)
        pre_softmax = F.softmax(features, dim=1)
        threshold = 0.7
        prediction = pre_softmax > threshold

        loss_arr = torch.zeros(len(features))
        iou_arr = torch.zeros(len(features))
        dice_arr = torch.zeros(len(features))
        for i in range(len(features)):
            loss_arr[i] = cal_loss(features[i], label[i])
            iou_arr[i], dice_arr[i] = (prediction[i], label[i])
        loss = torch.mean(loss_arr)
        iou = torch.mean(iou_arr)
        dice = torch.mean(dice_arr)

        valid_acc.update(dice_coeff(prediction, label), prediction.size(0))
        valid_loss.update(loss.data.item(), prediction.size(0))
        valid_iou.update(iou)
        valid_dice.update(dice)

        if idx == 0:
            logger.log_images(img.cpu(), label.cpu(), features.cpu(), step)
    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f} Iou:{:.4e} Dice:{:.4e}".format(valid_loss.avg, valid_acc.avg, valid_iou.avg, valid_dice.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg, "valid_iou": valid_iou, "valid_dice": valid_dice}


def get_cuda_info():
    # 检查GPU是否可用
    if torch.cuda.is_available():
        # 获取当前GPU设备
        device = 'cuda:3'

        # 获取GPU设备的总显存量和已使用的显存量
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)

        # 计算可用的显存量
        available_memory = total_memory - allocated_memory - cached_memory

        # 打印显存信息
        print(f"Total GPU memory: {total_memory} bytes")
        print(f"Allocated GPU memory: {allocated_memory} bytes")
        print(f"Cached GPU memory: {cached_memory} bytes")
        print(f"Available GPU memory: {available_memory} bytes")
    else:
        print("No GPU available.")


def cal_loss(pre, target):
    pre = torch.squeeze(pre)
    bce = F.binary_cross_entropy_with_logits(pre, target)
    return bce


def metric(pred_mask, true_mask):
    pred_mask = torch.squeeze(pred_mask)
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = intersection.float() / union.float()
    dice = (2 * intersection.float()) / (pred_mask.sum() + true_mask.sum())
    return iou, dice


def save_checkpoint(state, is_best, root, filename='checkpoints.pth.tar'):
    if not os.path.exists(root):
        os.mkdir(root)
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


if __name__ == '__main__':
    main()
