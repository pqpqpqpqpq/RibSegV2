import torch
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def metric():
    file_path = 'E:/Data/prediction/ribSegV2'
    files = os.listdir(file_path)
    ious = torch.zeros(len(files))
    dices = torch.zeros(len(files))
    i = 0
    for file in files:
        file_folder = os.path.join(file_path, file)
        prediction_path = os.path.join(file_folder, 'prediction.nii.gz')
        msk_path = os.path.join(file_folder, 'image.nii.gz')

        pre = sitk.ReadImage(prediction_path)
        pre = torch.from_numpy(sitk.GetArrayFromImage(pre))
        pre = (pre > 4).to(torch.int8)
        msk = sitk.ReadImage(msk_path)
        msk = torch.from_numpy(sitk.GetArrayFromImage(msk))
        msk = (msk > 0).to(torch.int8)
        iou, dice = cal_iou_dice(pre, msk)
        print('index:{} iou:{} dice:{}'.format(i, iou, dice))
        ious[i] = iou
        dices[i] = dice
        i += 1
    print('IOU:{} DICE:{}'.format(torch.mean(ious), torch.mean(dices)))

def cal_iou_dice(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = intersection.float() / union.float()
    dice = (2 * intersection.float()) / (pred_mask.sum() + true_mask.sum())
    return iou, dice


def draw_histogram(data, label):
    bins = 10
    plt.hist(data, bins=bins, alpha=0.5, label=label)

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

if __name__ == '__main__':
    metric()
    # iou1 = [0.5033422, 0.56525636, 0.61441493, 0.6206999, 0.62944925,
    #         0.6400049, 0.659828, 0.6603163, 0.66084886, 0.66684836,
    #         0.6761579, 0.6775763, 0.68653953, 0.6876688, 0.691097,
    #         0.69163084, 0.69169253, 0.7140038, 0.71413547, 0.7169507,
    #         0.71703726, 0.7171691, 0.7176373, 0.7317647, 0.7341689,
    #         0.73491985, 0.73584616, 0.744827, 0.74610704, 0.7462012,
    #         0.74696535, 0.74976885, 0.7530702, 0.75369984, 0.7551221,
    #         0.75844836, 0.7605861, 0.76071924, 0.7635428, 0.7637106,
    #         0.76490474, 0.76557016, 0.7708034, 0.77110386, 0.77393556,
    #         0.77540153, 0.7765157, 0.7773359, 0.78349286, 0.7835265,
    #         0.78506124, 0.78891337, 0.78911555, 0.7898942, 0.7903474,
    #         0.7916709, 0.79192173, 0.7920705, 0.79324824, 0.7969967,
    #         0.79971814, 0.80065906, 0.80091286, 0.8010942, 0.80136025,
    #         0.802018, 0.8045454, 0.8083431, 0.8237181, 0.84089786]
    # iou3 = [0.36091065, 0.38990757, 0.39220348, 0.40029013, 0.4174641,
    #         0.43237028, 0.5184784, 0.56134695, 0.57324374, 0.5748423,
    #         0.60841155, 0.6092061, 0.6139427, 0.6174664, 0.6225244,
    #         0.6356722, 0.6390753, 0.63936776, 0.6406239, 0.65368366,
    #         0.6591533, 0.6720041, 0.67556834, 0.68171144, 0.6822541,
    #         0.6836092, 0.6854717, 0.6893387, 0.6931875, 0.6968306,
    #         0.69906753, 0.70040226, 0.7014057, 0.7028296, 0.7071447,
    #         0.7079536, 0.7098201, 0.7115406, 0.71424836, 0.7155782,
    #         0.7174737, 0.7176082, 0.72088814, 0.72257125, 0.72720426,
    #         0.7281381, 0.7285672, 0.7318153, 0.7371239, 0.73855245,
    #         0.73883164, 0.73937947, 0.74142545, 0.7439462, 0.74501705,
    #         0.7500523, 0.7527087, 0.75481653, 0.75498885, 0.7552065,
    #         0.755927, 0.7583375, 0.7585891, 0.7586301, 0.7591704,
    #         0.75987166, 0.76217216, 0.76232976, 0.76959485, 0.79874027]
    #
    # dice1 = [0.66963091, 0.72225401, 0.7611611, 0.76596521, 0.77259141,
    #          0.78049141, 0.79505587, 0.79541024, 0.79579652, 0.80013079,
    #          0.80679499, 0.80780384, 0.81413986, 0.81493336, 0.81733575,
    #          0.81770893, 0.81775206, 0.83314145, 0.83323107, 0.83514418,
    #          0.83520293, 0.83529233, 0.83560983, 0.84510867, 0.84670977,
    #          0.84720901, 0.84782418, 0.85375453, 0.85459485, 0.85465662,
    #          0.85515761, 0.8569919, 0.85914438, 0.85955396, 0.86047815,
    #          0.86263366, 0.86401466, 0.86410057, 0.8659192, 0.86602711,
    #          0.86679436, 0.86722145, 0.87056912, 0.87076073, 0.87256334,
    #          0.87349426, 0.87420081, 0.87472031, 0.87860497, 0.87862612,
    #          0.87959026, 0.88200287, 0.88212923, 0.88261554, 0.88289837,
    #          0.88372359, 0.88387981, 0.88397249, 0.88470544, 0.88703189,
    #          0.88871487, 0.88929555, 0.88945211, 0.88956391, 0.88972793,
    #          0.89013317, 0.89168762, 0.89401519, 0.90333926, 0.91357364]
    # dice3 = [0.53039581, 0.5610554, 0.5634284, 0.57172458, 0.58902953,
    #          0.60371298, 0.68289202, 0.71905473, 0.72874117, 0.73003153,
    #          0.75653716, 0.75715109, 0.76079864, 0.76349825, 0.76735289,
    #          0.77726111, 0.77979979, 0.78001747, 0.78095157, 0.79057884,
    #          0.79456588, 0.8038307, 0.80637515, 0.81073534, 0.81111894,
    #          0.81207586, 0.81338856, 0.81610479, 0.8187959, 0.82133192,
    #          0.82288374, 0.82380774, 0.82450141, 0.82548437, 0.82845313,
    #          0.82900797, 0.8302863, 0.83146213, 0.8333079, 0.8342123,
    #          0.8354989, 0.83559011, 0.8378094, 0.83894498, 0.84205937,
    #          0.84268506, 0.84297235, 0.84514244, 0.84867167, 0.84961769,
    #          0.84980237, 0.85016464, 0.85151558, 0.85317561, 0.85387938,
    #          0.857177, 0.85890904, 0.8602797, 0.86039163, 0.86053296,
    #          0.86100052, 0.86256192, 0.86272466, 0.86275119, 0.86310046,
    #          0.8635535, 0.86503712, 0.86513862, 0.8697978, 0.88811073]
    # draw_histogram(iou1, 'iou1')
    # draw_histogram(iou3, 'iou3')
    # plt.legend(loc='upper left')
    # plt.show()
    # draw_histogram(dice1, 'dice1')
    # draw_histogram(dice3, 'dice3')
    # plt.legend(loc='upper left')
    # plt.show()