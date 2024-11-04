import numpy as np
import pandas as pd
import os
import time
import torch
from tqdm import tqdm
import SimpleITK as sitk
# from sklearn.metrics import roc_auc_score
from torchvision.utils import make_grid
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch.nn.functional as F
import csv

from utils.image_utils import calc_dice


def update_tensorboard(metrics, epoch, tensorboard_writer, do_validation=True):
    for phase in metrics:
        if phase != 'val' or do_validation:
            # scalar_dict = {}
            for key in metrics[phase]:
                if len(metrics[phase][key]) > 0 and key != 'epoch':
                    # scalar_dict[key] = metrics[phase][key][-1]
                    tensorboard_writer.add_scalar(phase + '_' + key, metrics[phase][key][-1], epoch)
            # tensorboard_writer.add_scalars(phase, scalar_dict, epoch)


def update_tensorboard_image(feat_dict, global_step, tensorboard_writer):
    for i, key in enumerate(feat_dict):
        feat_map = torch.mean(feat_dict[key], dim=1)
        # 特征图按通道维度取均值
        feat_size = feat_map.size()
        slice_id = feat_size[1] // 2
        # 取特征图中间层(Z方向)进行可视化
        tensorboard_writer.add_image(key, make_grid(feat_map[:, slice_id, :, :],
                                                    padding=20, normalize=True,
                                                    scale_each=True, pad_value=1), global_step)


def analysis_train_output(outputs_dict, epoch_results_dict, phase='train'):
    """
    获取不同模型对应的log_string,按照网络输出填写result_dict
    :param outputs_dict:
    :param epoch_results_dict:
    :param phase:
    :return:
    """
    sum_tags = ['tp', 'tn', 'fp', 'fn']
    for key in outputs_dict:
        if key not in epoch_results_dict[phase]:
            epoch_results_dict[phase][key] = []
        if key not in sum_tags:
            epoch_results_dict[phase][key].append(outputs_dict[key].mean().item())
        else:
            epoch_results_dict[phase][key].append(outputs_dict[key].sum().item())

    exclude_tags = ['learning_rate', 'epoch'] + ['tp', 'tn', 'fp', 'fn']
    log_string = ' '.join([key + ': %0.2f' % epoch_results_dict[phase][key][-1] for key in epoch_results_dict[phase]
                           if key not in exclude_tags])

    loss = outputs_dict['loss'].mean()
    return loss, log_string


def update_metrics(monitor_metrics, epoch_results_dict):
    # train
    exclude_tags = ['tp', 'tn', 'fp', 'fn']
    for phase in epoch_results_dict:
        for key in epoch_results_dict[phase]:
            if key not in monitor_metrics[phase]:
                monitor_metrics[phase][key] = []
            if key not in exclude_tags:
                monitor_metrics[phase][key].append(np.mean(epoch_results_dict[phase][key]))  # 保留均值
        if 'tp' in epoch_results_dict[phase] and 'tn' in epoch_results_dict[phase] and 'fn' in epoch_results_dict[
            phase] and 'fp' in epoch_results_dict[phase]:
            tp = np.sum(epoch_results_dict[phase]['tp'])
            tn = np.sum(epoch_results_dict[phase]['tn'])
            fn = np.sum(epoch_results_dict[phase]['fn'])
            fp = np.sum(epoch_results_dict[phase]['fp'])
            recall = float(tp) / float(tp + fn)
            precision = float(tp) / float(tp + fp)
            accuracy = float(tp + tn) / float(tp + fp + tn + fn)
            if 'recall' not in monitor_metrics[phase]:
                monitor_metrics[phase]['recall'] = []
            if 'precision' not in monitor_metrics[phase]:
                monitor_metrics[phase]['precision'] = []
            if 'accuracy' not in monitor_metrics[phase]:
                monitor_metrics[phase]['accuracy'] = []
            monitor_metrics[phase]['recall'].append(recall)
            monitor_metrics[phase]['precision'].append(precision)
            monitor_metrics[phase]['accuracy'].append(accuracy)



def write_csv(csv_path, content, mul=True, mod="a"):
    with open(csv_path, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)


def analysis_pneumonia_cls_seg(cf, batch_dict, result_dict, output_dir):
    # have_gt_flag = True
    # save_visual = False
    images = batch_dict['inputs']
    # gt_mask = batch_dict['mask']
    # mask_label = batch_dict['mask_label']
    label = batch_dict['label']
    # infos = batch_dict['infos']

    predict_cls = result_dict['predict_cls']
    predict_seg = result_dict['predict_seg']

    # class
    output_csv_path = os.path.join(output_dir, 'test_class.csv')
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['seriesuid', 'gt0', 'gt1', 'gt2', 'prob0', 'prob1', 'prob2']], mod='w')
    for i in range(images.size(0)):
        sid = infos['seriesuid'][i]
        gt_tmp = np.array(label.cpu())[i]
        if hasattr(cf, 'cls_output_map') and cf.cls_output_map == 'Sigmoid':
            prob_tmp = np.array(F.sigmoid(predict_cls).cpu())[i]
        else:
            prob_tmp = np.array(F.softmax(predict_cls, dim=1).cpu())[i]
        write_csv(output_csv_path, [[sid, gt_tmp[0], gt_tmp[1], gt_tmp[2], prob_tmp[0], prob_tmp[1], prob_tmp[2]]], mod='a')

    # seg
    if len(gt_mask.size()) == 4:
        size = (gt_mask.size(1), gt_mask.size(2), gt_mask.size(3))
    else:
        size = (gt_mask.size(2), gt_mask.size(3), gt_mask.size(4))
    predict_seg = F.upsample(input=predict_seg,
                                 size=size,
                                 mode='trilinear')
    output_csv_path = os.path.join(output_dir, 'seg.csv')
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['seriesuid',] + ths], mod='w')
    for i in range(images.size(0)):
        sid = infos['seriesuid'][i]
        mask_label_tmp = np.array(mask_label.cpu())[i]
        if mask_label_tmp != 1:
            continue
        gt_mask_tmp = np.array(gt_mask.cpu())[i]
        predict_seg_tmp = np.array(F.sigmoid(predict_seg.cpu()))[i][0]
        dices = []
        for th in ths:
            predict_seg_th = predict_seg_tmp > th
            dices.append(calc_dice(gt_mask_tmp != 0, predict_seg_th))
        write_csv(output_csv_path, [[sid] + dices], mod='a')

        # image = np.array(batch_dict['inputs'][i].cpu()).squeeze()
        # image_sitk = sitk.GetImageFromArray(image)
        # sitk.WriteImage(image_sitk, sid+'_img.nii.gz')
        #
        # pre_sitk = sitk.GetImageFromArray(predict_seg_tmp)
        # sitk.WriteImage(pre_sitk, sid+'_pre.nii.gz')


def analysis_pneumonia_cls_seg_two_cls(cf, batch_dict, result_dict, output_dir):
    # have_gt_flag = True
    # save_visual = False
    images = batch_dict['inputs']
    gt_mask = batch_dict['mask']
    mask_label = batch_dict['mask_label']
    label = batch_dict['label']
    infos = batch_dict['infos']

    predict_cls = result_dict['predict_cls']
    predict_seg = result_dict['predict_seg']

    # class
    output_csv_path = os.path.join(output_dir, 'test_class.csv')
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['seriesuid', 'gt0', 'gt1', 'prob0', 'prob1']], mod='w')
    for i in range(images.size(0)):
        sid = infos['seriesuid'][i]
        gt_tmp = np.array(label.cpu())[i]
        if hasattr(cf, 'cls_output_map') and cf.cls_output_map == 'Sigmoid':
            prob_tmp = np.array(F.sigmoid(predict_cls).cpu())[i]
        else:
            prob_tmp = np.array(F.softmax(predict_cls, dim=1).cpu())[i]
        write_csv(output_csv_path, [[sid, gt_tmp[0], gt_tmp[1], prob_tmp[0], prob_tmp[1]]], mod='a')

    # seg
    if len(gt_mask.size()) == 4:
        size = (gt_mask.size(1), gt_mask.size(2), gt_mask.size(3))
    else:
        size = (gt_mask.size(2), gt_mask.size(3), gt_mask.size(4))
    predict_seg = F.upsample(input=predict_seg,
                                 size=size,
                                 mode='trilinear')
    output_csv_path = os.path.join(output_dir, 'seg.csv')
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['seriesuid',] + ths], mod='w')
    for i in range(images.size(0)):
        sid = infos['seriesuid'][i]
        mask_label_tmp = np.array(mask_label.cpu())[i]
        if mask_label_tmp != 1:
            continue
        gt_mask_tmp = np.array(gt_mask.cpu())[i]
        predict_seg_tmp = np.array(F.sigmoid(predict_seg.cpu()))[i][0]
        dices = []
        for th in ths:
            predict_seg_th = predict_seg_tmp > th
            dices.append(calc_dice(gt_mask_tmp != 0, predict_seg_th))
        write_csv(output_csv_path, [[sid] + dices], mod='a')

        # image = np.array(batch_dict['inputs'][i].cpu()).squeeze()
        # image_sitk = sitk.GetImageFromArray(image)
        # sitk.WriteImage(image_sitk, sid+'_img.nii.gz')
        #
        # pre_sitk = sitk.GetImageFromArray(predict_seg_tmp)
        # sitk.WriteImage(pre_sitk, sid+'_pre.nii.gz')


def analysis_seg(batch_dict, result_dict, output_dir):
    images = batch_dict['inputs']
    gt_mask = batch_dict['mask']
    infos = batch_dict['infos']

    predict_seg = result_dict['predict_seg']
    predict_seg = F.upsample(input=predict_seg,
                                 size=(gt_mask.size(2), gt_mask.size(3), gt_mask.size(4)),
                                 mode='trilinear')

    # seg
    output_csv_path = os.path.join(output_dir, 'seg.csv')
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['seriesuid', ] + ths], mod='w')
    for i in range(images.size(0)):
        sid = infos['seriesuid'][i]
        gt_mask_tmp = np.array(gt_mask.cpu())[i].squeeze()
        predict_seg_tmp = np.array(F.sigmoid(predict_seg.cpu()))[i][0]
        # predict_seg_tmp = predict_seg_tmp > 0.4
        dices = []
        for th in ths:
            predict_seg_th = predict_seg_tmp > th
            dices.append(calc_dice(gt_mask_tmp != 0, predict_seg_th))
        # dice = calc_dice(gt_mask_tmp!=0, predict_seg_tmp)
        write_csv(output_csv_path, [[sid] + dices], mod='a')

        image = np.array(images[i].cpu()).squeeze()
        image_sitk = sitk.GetImageFromArray(image)
        sitk.WriteImage(image_sitk, os.path.join(output_dir, sid+'_img.nii.gz'))

        gt_sitk = sitk.GetImageFromArray(gt_mask_tmp.astype(np.uint8))
        sitk.WriteImage(gt_sitk, os.path.join(output_dir, sid + '_gt.nii.gz'))

        pre_sitk = sitk.GetImageFromArray((predict_seg_tmp>0.5).astype(np.uint8))
        sitk.WriteImage(pre_sitk, os.path.join(output_dir, sid + '_pre.nii.gz'))


def analysis_test_output(cf, batch_dict, result_dict):
    if not os.path.exists(cf.output_files_dir):
        os.mkdir(cf.output_files_dir)
    output_dir = cf.output_files_dir

    if 'pneumonia_seg_cls' == os.path.basename(cf.exp_source) or 'pneumonia_cls_common' == os.path.basename(cf.exp_source) :
        # analysis_pneumonia_cls_seg(cf, batch_dict, result_dict, output_dir)
        analysis_pneumonia_cls_seg_two_cls(cf, batch_dict, result_dict, output_dir)

    if 'pneumonia_seg' == os.path.basename(cf.exp_source):
        analysis_seg(batch_dict, result_dict, output_dir)

    if 'covid_19_seg_benchmark' in os.path.basename(cf.exp_source):
        analysis_seg(batch_dict, result_dict, output_dir)



    # if 'pneumonia_seg' in cf.exp_source:
    #     output_dir = os.path.join(cf.output_files_dir, '1806')
    #     if not os.path.exists(output_dir):
    #         os.mkdir(output_dir)
    #     # output_dir = cf.output_files_dir
    #     output_csv_path = os.path.join(output_dir, '1806.csv')
    #     if not os.path.exists(output_csv_path):
    #         write_csv(output_csv_path, [['seriesuid', 'dice']], mod='w')
    #     for batch_dict, result_dict in analysis_list:
    #         infos = batch_dict['infos']
    #         predicts = result_dict['predict']
    #         gts = batch_dict['gt']
    #         images = batch_dict['inputs']
    #
    #         seriesuids = infos['seriesuid']
    #         for idx, sid in enumerate(seriesuids):
    #             gt = gts[idx].cpu().numpy()
    #             if cf.num_seg_classes == 1:
    #                 predict = F.sigmoid(predicts[idx]).cpu().numpy()
    #                 predict = predict > 0.5
    #                 dice = calc_dice(gt, predict)
    #                 write_csv(output_csv_path, [[sid, dice]], mod='a')
    #
    #                 output_sitk = sitk.GetImageFromArray(predict.squeeze().astype(np.uint8))
    #                 sitk.WriteImage(output_sitk, os.path.join(output_dir, sid + '_predict.nii.gz'))
    #
    #                 gt_mask = gt.squeeze()
    #                 gt_sitk = sitk.GetImageFromArray(gt_mask.astype(np.uint8))
    #                 sitk.WriteImage(gt_sitk, os.path.join(output_dir, sid + '_gt_mask.nii.gz'))
    #
    #             else:
    #                 predict = F.softmax(predicts[idx], dim=0).cpu().numpy()
    #                 predict = predict > 0.5
    #                 dice = calc_dice(gt[1:], predict[1:])
    #                 write_csv(output_csv_path, [[sid, dice]], mod='a')
    #
    #                 output_sitk = sitk.GetImageFromArray(predict[1:].squeeze().astype(np.uint8))
    #                 sitk.WriteImage(output_sitk, os.path.join(output_dir, sid + '_predict.nii.gz'))
    #
    #                 gt_mask = np.argmax(gt, axis=0)
    #                 gt_sitk = sitk.GetImageFromArray(gt_mask.astype(np.uint8))
    #                 sitk.WriteImage(gt_sitk, os.path.join(output_dir, sid + '_gt_mask.nii.gz'))
    #
    #             image = images[idx].cpu().numpy().squeeze()
    #             image_sitk = sitk.GetImageFromArray(image)
    #             sitk.WriteImage(image_sitk, os.path.join(output_dir, sid + '_img.nii.gz'))
    #
    #     # df_output.to_csv(os.path.join(output_dir,'1777.csv'), index=False)
