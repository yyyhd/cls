# encoding=utf-8
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import cv2
from PIL import Image
import time
import random
import glob
import torch
import deepdish as dd
import matplotlib.pyplot as plt
from utils.image_utils import normalize_image, CreateTransformer
# from utils.image_utils_2D import normalize_image, CreateTransformer
from utils.logger_utils import NoneLog




def resize_moving(image_zyx, mask=None):
    '''
    Return a resized (of size out_size) npy image.
    '''

 
    new_shape = (8, 80, 80)
    origin_shape = np.array(image_zyx.shape, dtype=float)
    shape_scale = new_shape / origin_shape
    # image_zyx = zoom(image_zyx, shape_scale, order=0)#......................................................

    return image_zyx

def resize_moving_2D(image_yx, mask=None):
    '''
    Return a resized (of size out_size) npy image.
    '''

    new_shape = (34, 34)
    origin_shape = np.array(image_yx.shape, dtype=float)
    shape_scale = new_shape / origin_shape
    image_zyx = zoom(image_yx, shape_scale, order=0)

    return image_zyx

def get_lung_offset(lungmask_arr, spacing_zyx, extend_mm=16):
    lung_mask_l = lungmask_arr == 1
    lung_mask_r = lungmask_arr == 2
    lung_mask_b = (lung_mask_l + lung_mask_r) > 0
    zz, yy, xx = np.where(lung_mask_b)

    # extend 16 mm.
    extend_zyx = np.round(extend_mm / np.array(spacing_zyx)).astype(int)
    zz, yy, xx = np.array(zz), np.array(yy), np.array(xx)
    voi_zz = [max(0, zz.min() - extend_zyx[0]), zz.max() + extend_zyx[0] + 1]
    voi_yy = [max(0, yy.min() - extend_zyx[1]), yy.max() + extend_zyx[1] + 1]
    voi_xx = [max(0, xx.min() - extend_zyx[2]), xx.max() + extend_zyx[2] + 1]

    recommed_offset_zyxzyx = [voi_zz[0], voi_yy[0], voi_xx[0], voi_zz[1], voi_yy[1], voi_xx[1]]
    return recommed_offset_zyxzyx


class DataCustom(Dataset):

    def __init__(self, cf, logger=None, phase='train'):
        assert phase in ['train', 'val', 'test'], "phase must be one of 'train', 'val' and 'test'."
        if logger is None:
            logger = NoneLog()
        self.phase = phase
        self.cf = cf
        self.input_size = cf.input_size

        # from result.thyroid_common_npz.configs import configs as cf

        if phase != 'test':
            data_paths_list = []
            cls_csv_dir = os.path.join(cf.cls_csv_dir, phase)
            csv_paths = glob.glob(os.path.join(cls_csv_dir, '*.csv'))
            df_cls = pd.DataFrame()
            for csv_path in csv_paths:
                df_cls = df_cls.append(pd.read_csv(csv_path))
            data_paths = cf.data_path
            for idx, item in df_cls.iterrows():
                data_path = os.path.join(data_paths, item[0])
                if os.path.exists(data_path):
                    data_paths_list.append(data_path)

            if cf.black_list:
                black_list = pd.read_csv(cf.black_list)['seriesuid'].tolist()
                df_cls = df_cls[[True if sid not in black_list else False for sid in df_cls['seriesuid'].tolist()]]
            # logger.info(phase + ' label0 num: %s, label1 num: %s, label2 num: %s' % (sum(df_cls['label'] == 0), sum(df_cls['label'] == 1), sum(df_cls['label'] == 2)))
            self.data_paths_list = data_paths_list


        if phase == 'test':
            data_paths_list = []

            cls_csv_dir = os.path.join(cf.cls_csv_dir, phase)
            csv_paths = glob.glob(os.path.join(cls_csv_dir, '*.csv'))
            df_cls = pd.DataFrame()
            for csv_path in csv_paths:
                df_cls = df_cls.append(pd.read_csv(csv_path))
            data_paths = cf.data_path
            for idx, item in df_cls.iterrows():
                data_path = os.path.join(data_paths, item[0])
                if os.path.exists(data_path):
                    data_paths_list.append(data_path)
            self.data_paths_list = data_paths_list

    def __len__(self):
        return len(self.data_paths_list)

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))

        if self.phase != 'test':
            h5_path = self.data_paths_list[idx]
            #print(h5_path)
            # h5_file = dd.io.load(h5_path)#h5
            h5_file = np.load(h5_path)  # npz

            if h5_file['label'] == 2 :
                label = np.array([1, 0])
            if  h5_file['label'] == 0 or h5_file['label'] == 1:
                label = np.array([0, 1])
            
            #print(h5_file.files,'h5')
            image = h5_file['nodule_voi_zyx']
            
            image = image.astype(np.float32)
           
            image = resize_moving(image)
            newimg = normalize_image(image, ntype='normalize')
           

            if self.phase == 'train':
                transformer = CreateTransformer(self.cf, random_scale=self.cf.da_kwargs['do_scale'],
                                                random_crop=self.cf.da_kwargs['random_crop'],
                                                random_flip=self.cf.da_kwargs['random_flip'])

            elif self.phase == 'val':
                transformer = CreateTransformer(self.cf, random_scale=False,
                                                random_crop=False,
                                                random_flip=False)

            newimg = transformer.image_transform_with_bbox(newimg, mask=None, pad_value=0,centerd=None, bbox=None)
          

            newimg = newimg.astype(np.float32)
          
            newimg = newimg[np.newaxis, ...].copy()
            
            patentid = h5_file['patientid'].tolist()
          
            
           
            return {'inputs': newimg, 'label': label, 'patentid': patentid}
           
        else:

            h5_path = self.data_paths_list[idx]
            # h5_file = dd.io.load(h5_path)
            h5_file = np.load(h5_path)  # npz
            # pneumonia_type = h5_file['annot']['pneumonia_type']
            #
            # spacing_zyx =  h5_file['dcm_tag']['spacing_zyx']

            if h5_file['label'] == 2 :
                label = np.array([1, 0])
            if  h5_file['label'] == 0 or h5_file['label'] == 1:
                label = np.array([0, 1])

            #print(h5_file.files,'h5')
            image = h5_file['nodule_voi_zyx']            # patenid = h5_path[50:57]
            # # patenid = h5_path[49:56]
            # if '_' in patenid:
            #     patenid = patenid.replace('_', '')

            # image = h5_file['nodule_voi_zyx']
            image = image.astype(np.float32)
            # lungmask_arr = h5_file['infer']['lung_mask']
            # bbox = get_lung_offset(lungmask_arr, spacing_zyx)
            # image = image[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

            # image = resize_moving_2D(image)
            image = resize_moving(image)
            # label = np.array([0, 0, 0])
            # label[pneumonia_type[0]-1] = 1

            image = normalize_image(image, ntype='normalize')

            

            transformer = CreateTransformer(self.cf, random_scale=False,
                                            random_crop=False,
                                            random_flip=False)

            image = transformer.image_transform_with_bbox(image, pad_value=0,
                                                          centerd=None, bbox=None)

            # mask = np.zeros([1, image.shape[0], image.shape[1], image.shape[2]], dtype=np.float32)
            # mask_label = 0
            newimg = image.astype(np.float32)
            
            image = newimg[np.newaxis, ...].copy()

            # sid = os.path.basename(h5_path).replace('.h5', '')
            # infos = {'seriesuid': sid}
            # label_cli = h5_file['label_cli']###clinical
            patentid = h5_file['patientid'].tolist()
            # patentid = h5_file['studyid'].tolist()
            return {'inputs': image, 'label': label, 'patientid': patentid}
            # return {'inputs': image, 'label': label, 'label_cli':label_cli.astype(np.float32), 'patentid': patentid}
            # return {'inputs': image, 'label': label}



