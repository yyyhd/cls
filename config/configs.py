# coding=utf-8
import os
import numpy as np
from yacs.config import CfgNode as CN

from config.default_configs import DefaultConfigs

class configs(DefaultConfigs):
    # ============ model setting ============
   
    
    model = 'model/resnetgn'


    input_channel = 1
    num_classes = 2

    cls_output_map = 'Sigmoid'  # one of 'Softmax' or 'Sigmoid'

    input_size2 = (5, 34, 34)



    input_size = (8, 80, 80)

    # data augment
    da_kwargs = {
        'do_elastic_deform': False,
        'alpha': (0., 1500.),
        'sigma': (30., 50.),
        'do_rotation': False,
        'angle_x': (0., 2 * np.pi),
        'angle_y': (0., 0),  # must be 0!!
        'angle_z': (0., 0),
        'do_scale': True,
        'scale': (0.9, 1.1),
        'random_crop': True,
        'random_flip': True,
    }


   



    # ============ I/O ============
    # dce
    cls_csv_dir = './csv_dir'

    data_path = './output/cut_dce'
 
    black_list = ''
    result_csv_path = './output/result/test_class.csv'  
    

   
    # ============ train ============

  
    batch_size = 8 #16
    n_workers = 16 # 16
    do_validation = True

    num_epochs =200
    weight_decay = 1e-3
    best_model_min_save_thresh = 20
    save_n_best_models = 5   # 5
    save_model_per_epochs = 1  # 5
    # one of 'SGD' or 'Adam. RMSprop
    optimizer = 'Adam'

    momentum = 0.9
    # LR = 0.0001
    default_lr = 1e-5
    learning_rate = []
    for i in range(num_epochs):
        if i < num_epochs * 0.5:   # 
            learning_rate.append(default_lr)
        elif i < num_epochs * 0.8:  #  
            learning_rate.append(default_lr * 0.1)
        else:
            learning_rate.append(default_lr * 0.01)

    # ============ test ============
    test_batch_size = 1
    is_convert_to_world_coord = False


    def __init__(self):
        super(configs, self).__init__()
        pass


