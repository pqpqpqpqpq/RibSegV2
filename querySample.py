import os
import random
import torch
import numpy as np
from tqdm import tqdm
from data.data import read_img, set_window, crop_cube, img_label_preprocess, img_preprocess_aug, img_mask_process_aug, QUEUE, HEIGHT, WEIGHT, bone_thres,connect_check,\
    del_surplus_mask, del_surplus_noMask, post_process
import SimpleITK as sitk
import random
# 设置随机种子（可选）
random.seed(42)  # 用任意整数作为种子
peth = 0.6
entropy_epoch = 150


class QuerySample:
    def __init__(self, model, epoch, args):
        self.model = model
        self.epoch = epoch
        self.args = args
        self.valid_gpu = args.gpu

        # AL
        self.random_init = ((epoch == 0) and (not args.domainAdaptation))
        self.random_num = args.random_num
        self.select_num = args.select_num

        # PRA
        self.PRA = args.PRA
        self.peth = args.peth
        self.save_entropy_map = args.get("save_entropy_map", True)

        # ST
        self.ST = args.ST
        self.self_train_begin_epoch = args.self_train_begin_epoch
        self.confident_thre = args.confident_thre
        self.save_pseudo_label = args.get("save_pseudo_label", False)

        # save folder and log
        self.unannotated_data_path = os.path.join(args.train_data)
        self.label_path = args.train_cube_path
        self.unannotated_data_list = os.listdir(args.train_data)
        self.selected_data_path = os.path.join(os.path.dirname(args.train_data), 'selected_data',
                                               int(epoch // args.active_iter))
        self.select_log = os.path.join(os.path.dirname(args.train_data),
                                       'select_log_{}.txt'.format(int(epoch // args.active_iter)))
        self.entropy_map_path = os.path.join(os.path.dirname(args.train_data), 'entropy_map',
                                             int(epoch // args.active_iter))
        self.pseudo_label_path = os.path.join(os.path.dirname(args.train_data), 'pseudo_label',
                                              int(epoch // args.active_iter))
        self.train_cube_path = args.train_cube_path

    def init_params_and_folder(self):
        if not os.path.exists(self.selected_data_path):
            os.makedirs(self.selected_data_path)
        if not os.path.exists(self.entropy_map_path):
            os.makedirs(self.entropy_map_path)
        if not os.path.exists(self.pseudo_label_path):
            os.makedirs(self.pseudo_label_path)
        self.select_log = open(self.select_log, 'w')

    def _get_high_entropy_mask(self, feature_map, img_name):
        pre_softmax = torch.softmax(feature_map, -1)
        logist = torch.log(pre_softmax)
        entropy_map = -torch.mul(pre_softmax, logist)
        entropy_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min())
        high_entropy_mask = (entropy_map > peth).to(torch.int)
        if self.save_entropy_map:
            label = high_entropy_mask.cpu().detach().numpy()
            label_nii = sitk.GetImageFromArray(label)
            sitk.WriteImage(label_nii,
                            os.path.join(self.entropy_map_path, img_name.split('-image.nii.gz')[0] + '-entropy.nii.gz'))
        return high_entropy_mask

    def _cal_uncertainty(self, prediction):
        count_0 = (prediction == 0).sum().item()
        count_1 = (prediction == 1).sum().item()
        count_2 = (prediction == 2).sum().item()
        count_3 = (prediction == 3).sum().item()
        count_4 = (prediction == 4).sum().item()
        count_5 = (prediction == 5).sum().item()
        count_6 = (prediction == 6).sum().item()
        count_7 = (prediction == 7).sum().item()
        count_8 = (prediction == 8).sum().item()
        count = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7
        count_sum = count_0 + count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8
        score = (0.1 * (count_1 + count_7) + 0.2 * (count_2 + count_6) + 0.3 * (count_3 + count_5) + 0.4 * (
            count_4)) / count_sum
        return score

    def _self_train(self, prediction, aug_img, img_name):
        # 将预测处理成cube
        prediction = (prediction > 3).cpu().to(torch.int8)
        pseudo_label = post_process(prediction)
        if self.save_pseudo_label:
            pseudo_label = sitk.GetImageFromArray(pseudo_label)
            img_name = img_name.split('-image.nii.gz')[0]
            sitk.WriteImage(pseudo_label, os.path.join(self.pseudo_label_path, img_name + '-prediction.nii.gz'))
        img_arr, mask_arr = img_label_preprocess(img=aug_img, label=pseudo_label, process=False)
        for i in range(len(img_arr)):
            cube_mask_path = os.path.join(self.train_cube_path, img_name, str(i))
            if not os.path.exists(cube_mask_path):
                os.makedirs(cube_mask_path)
            np.savez(os.path.join(cube_mask_path, 'img.npz'), img_arr[i])
            np.savez(os.path.join(cube_mask_path, 'msk.npz'), mask_arr[i])

    @torch.no_grad()
    def _get_prediction_feature_map(self, cube, shape):
        # 获取entropy
        prediction = torch.zeros(shape).cuda(self.args.gpu)
        feature_map = torch.zeros(shape).cuda(self.args.gpu)
        for img, position in cube:
            zmin, ymin, xmin, zmax, ymax, xmax = position
            img = torch.from_numpy(img).cuda(self.args.gpu).to(torch.float32)
            feature = self.model(x=img).squeeze()

            feature_map[zmin:zmax, ymin:ymax, xmin:xmax] = feature_map[zmin:zmax, ymin:ymax, xmin:xmax] + feature[:(
                        zmax - zmin), :(ymax - ymin), :(xmax - xmin)]
            cube_pre = (feature > 0).squeeze()
            prediction[zmin:zmax, ymin:ymax, xmin:xmax] = prediction[zmin:zmax, ymin:ymax,
                                                          xmin:xmax] + cube_pre[:(zmax - zmin),
                                                                       :(ymax - ymin), :(xmax - xmin)]
        yield prediction, feature_map

    def EGPA(self, select_samples):
        files = os.listdir(self.entropy_folder_path)
        index = 0
        for select_sample in select_samples:
            image_name = select_sample.split('-image.nii.gz')[0]
            high_entropy_mask_path = os.path.join(self.entropy_folder_path, image_name + '-entropy.nii.gz')
            high_entropy_mask = read_img(high_entropy_mask_path).to(torch.int8)
            img_path = os.path.join(self.selected_data_path, image_name + '-image.nii.gz')
            img = read_img(img)
            mask_path = os.path.join(self.label_path, image_name + '-rib-seg.nii.gz')
            mask = read_img(mask_path).astype(np.int8)

            aug_img, aug_msk = img_mask_process_aug(img, mask, high_entropy_mask)

            cubes = crop_cube(high_entropy_mask, (64, 64, 64))
            count = 0
            for (cube, position) in cubes:
                cube = torch.squeeze(torch.Tensor(cube))
                value, num = torch.unique(cube, return_counts=True)
                if len(num.numpy()) > 1:
                    # print(value, num)
                    if num.numpy()[-1] > 500:
                        zmin, ymin, xmin, zmax, ymax, xmax = position
                        if zmax - zmin == 64 and ymax - ymin == 64 and xmax - xmin == 64:
                            img_cube = aug_img[zmin:zmax, ymin:ymax, xmin:xmax]
                            mask_cube = aug_msk[zmin:zmax, ymin:ymax, xmin:xmax]
                            path = os.path.join(self.train_cube_path, image_name, str(count))
                            if not os.path.exists(path):
                                os.makedirs(path)
                            np.savez(os.path.join(path, 'img.npz'), img_cube)
                            np.savez(os.path.join(path, 'msk.npz'), mask_cube)
                            count += 1
            index += 1
            print('index:{} {}获取cube数:{}'.format(index, select_sample, count))

    def random_select_samples(self):
        self.select_log.write('epoch:{} 随机挑选初始样本\n'.format(self.epoch))
        selected_datas = random.sample(self.unannotated_data_list, self.random_num)
        for selected_data in selected_datas:
            self.select_log.write(selected_data + '\n')
            os.rename(os.path.join(self.unannotated_data_path, selected_data),
                      os.path.join(self.selected_data_path, selected_data))
        self.select_log.close()
        self.save_cube_label_nii(
            save_path=self.train_cube_path,
            img_path=self.selected_data_path,
            label_path=self.args.label_path,
            process=True
        )

    # Multi-Windows Committee Query
    @torch.no_grad()
    def MWCQ(self):
        self.select_log.write('epoch:{} 主动挑选初始样本：\n'.format(self.epoch))
        self.model.eval()
        confident_count = 0
        scores = []
        select_samples = []
        names = os.listdir(self.unannotated_data_path)
        for img_name in tqdm(names):
            cube, aug_img, img_name = self.img_process_aug(img_path=os.path.join(self.unannotated_data_path, img_name))
            prediction, feature_map = self._get_prediction_feature_map(cube, aug_img.shape)
            # 如果要做局部标注的话，保存熵图
            if self.PRA:
                high_entropy_mask = self._get_high_entropy_mask(feature_map, img_name)

            score = self._cal_uncertainty(prediction)

            # 如果启用self-train的话，保存高置信度样本的预测结果，并处理成训练用的cube
            if self.ST and score < self.confident_thre and self.epoch >= self.self_train_begin_epoch:
                pseudo_label = self._self_train(prediction, aug_img, img_name)
            else:
                scores.append(score)
                select_samples.append(img_name)
                self.select_log.write('{}: score:{}\n'.format(img_name, score))

        sorted_tuples = sorted(zip(scores, select_samples), key=lambda x: x[0], reverse=True)
        sorted_scores = [score for score, _ in sorted_tuples]
        sorted_select_samples = [img for _, img in sorted_tuples]
        select_samples = sorted_select_samples[:self.select_num]
        self.select_log.write('最终选择:{}, 得分:{}\n'.format(select_samples, sorted_scores[:self.select_num]))
        self.select_log.close()
        # 把数据从备选区放到训练区()
        for select_sample in select_samples:
            os.rename(os.path.join(self.unannotated_data_path, select_sample),
                      os.path.join(self.selected_data_path, select_sample))

        # 确定了最终选择的样本，我们再根据之前保存的熵图来裁剪关注区域
        if self.PRA:
            self.EGPA(select_samples)

    def query(self):
        if self.random_init:
            self.random_select_samples()
        else:
            self.MWCQ()