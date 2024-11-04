# encoding=utf-8
import argparse
import os
import time
import torch
import numpy as np
from torch import optim
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from utils.logger_utils import get_logger
import utils.evaluator_utils as eval_utils
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
from pandas import  DataFrame
import utils.exp_utils as util
import thyroid_common_npz.dataset as dataset
from model import resnetgn as model
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.cuda.current_device()
torch.cuda._initialized = True


def train(logger, cf):  # , model, dataset
    logger.info("performing training with model {}".format(cf.model))

    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    net = model.Net(cf, logger).cuda()

    print(cf.optimizer,'cf.optimizer')

    if cf.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay,
                              momentum=cf.momentum)
    elif cf.optimizer == 'Adam':
           optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    elif cf.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=cf.learning_rate[0], alpha=0.9, eps=1e-08, weight_decay=cf.weight_decay,
                                  momentum=cf.momentum, centered=False)
    else:
        optimizer = optim.Adam(net.parameters(), lr=cf.LR, weight_decay=cf.weight_decay)

    # prepare monitoring
    monitor_metrics = util.prepare_monitoring(cf)

    if cf.use_pretrain_model:
        util.load_checkpoint(net, cf.transfer_learning_weight_path, mode='pretrain', cf=cf)
        logger.info('load transfer learning weight {}'.format(cf.transfer_learning_weight_path))

    starting_epoch = 1
    if cf.resume_to_checkpoint is not False:
        resume_epoch, monitor_metrics = util.load_checkpoint(net, cf.resume_to_checkpoint, mode='resume', optimizer=optimizer)
        # print("1",monitor_metrics)

        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, resume_epoch))
        starting_epoch = resume_epoch + 1

    # 多GPU并行训练
    net = DataParallel(net)
    # net.cuda()

    # add this , can improve the train speed
    torch.backends.cudnn.benchmark = True

    logger.info('loading dataset and initializing batch generators...')
    data_file = dataset.DataCustom(cf, logger, phase="train")
    val_file = dataset.DataCustom(cf, logger, phase="val")
    target = []
    for i in data_file.data_paths_list:
    # for i in data_file.data_paths_list0: #dataset2
        file = np.load(i)
        if file["label"] == 1:
            target.append(1)
        if file["label"] == 0:
            target.append(0)
        

    target = np.array(target)
    print(target)
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])#报错 2024.1.23
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = torch.from_numpy(target).long()



    dataloaders = {phase: DataLoader(dataset.DataCustom(cf, logger, phase=phase),
                                     batch_size=cf.batch_size,
                                    #  shuffle={'train': False, 'val': False}[phase],
                                     shuffle={'train': sampler, 'val': None}[phase],                                 
                                     num_workers=cf.n_workers,
                                     pin_memory=True,
                                     drop_last=True)
                   for phase in ['train', 'val']}

    tensorboard_writer = SummaryWriter(cf.tensorboard_dir)

    lambda1 = lambda epoch:np.sin(epoch) / epoch
   
    for epoch in range(starting_epoch, cf.num_epochs + 1):
        epoch_start_time = toc_train = time.time()
        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]
            

        net.train()
        epoch_results_dict = {'train': {'learning_rate': [cf.learning_rate[epoch - 1]], 'epoch': [epoch]},
                              'val': {}}

        for batchidx, dataset_outputs in enumerate(dataloaders['train']):

            tic_fw = time.time()
            
            batch_outputs = net(dataset_outputs['inputs'], labels=dataset_outputs['label'], phase='train')
           

            loss, log_string = eval_utils.analysis_train_output(batch_outputs, epoch_results_dict, 'train')

            tic_bw = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info(
                'tr. batch {0}/{1} (ep. {2}) dl {3: .3f}/ fw {4:.3f}s / bw {5:.3f}s / total {6:.3f}s / lr {7} || {8}'.format(
                    batchidx + 1, len(dataloaders['train']), epoch, tic_fw - toc_train, tic_bw - tic_fw,
                    time.time() - tic_bw,
                    time.time() - tic_fw, optimizer.param_groups[-1]['lr'], log_string))

            torch.cuda.empty_cache()
            toc_train = time.time()


        train_time = time.time() - epoch_start_time

        with torch.no_grad():
            if cf.do_validation:
                logger.info("starting valiadation.")

                net.eval()
                
                for batchidx, dataset_outputs in tqdm(enumerate(dataloaders['val'])):

                    outputs = net(dataset_outputs['inputs'],labels=dataset_outputs['label'], phase='val')

                    eval_utils.analysis_train_output(outputs, epoch_results_dict, phase='val')

                    torch.cuda.empty_cache()


        # update monitoring and prediction plots
        eval_utils.update_metrics(monitor_metrics, epoch_results_dict)
        eval_utils.update_tensorboard(monitor_metrics, epoch, tensorboard_writer, cf.do_validation)
        util.model_select_save(cf, net, optimizer, monitor_metrics, epoch)
        # eval_utils.update_tensorboard_image(monitor_metrics, epoch, tensorboard_writer)

        epoch_time = time.time() - epoch_start_time
        logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(epoch, epoch_time, train_time,
                                                                                      epoch_time - train_time))
    tensorboard_writer.close()

def t_test(logger, cf):
    logger.info("performing testence with model {}".format(cf.model))

    net = model.Net(cf, logger)
    # net = model.Net2(cf, logger)

    try:
        util.load_checkpoint(net, cf.resume_to_checkpoint, mode='test')
        logger.info('resumed to checkpoint {}'.format(cf.resume_to_checkpoint))
    except Exception as e:
        logger.error('load checkpoint error! %s' % e)
        return None
    net = DataParallel(net)
    net.cuda()

    datasets = dataset.DataCustom(cf, logger, phase='test')

    test_data_loader = DataLoader(datasets,
                                  batch_size=cf.test_batch_size,
                                  shuffle=False,
                                  num_workers=cf.n_workers,
                                  pin_memory=True)
    i = 0
    if os.path.exists(cf.result_csv_path):
       os.remove(cf.result_csv_path)
    with torch.no_grad():
        for index, dataset_outputs in tqdm(enumerate(test_data_loader)):

            
            result_dict = net(dataset_outputs["inputs"], dataset_outputs["label"], dataset_outputs["patientid"], phase='test')
            eval_utils.analysis_thy(cf, dataset_outputs, result_dict, cf.result_csv_path, i)
            i = i+1
    print('over')
    #
    # eval_utils.save_to_csv(cf, output_dir, result_dict)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='train',
                        help='one out of : train / test / train_test')
    parser.add_argument('--exp_source', type=str, default='result/thyroid_common_npz', )  
    parser.add_argument('--exp_result', type=str, default='result/thyroid_common_npz', )
    parser.add_argument('--resume_to_checkpoint', action='store_true',
                        default=False,
                        help='False:不加载； True：加载last_state.pht； 路径：加载指定路径.')
    parser.add_argument('--checkpoint_path', type=str, default=False)  # 合并至上一个选项./result/thyroid_common_npz/epoch_model/model_600.pth
    parser.add_argument('--use_stored_settings', action='store_true', default=False,
                        help='load configs from existing stored_source instead of exp_source. always done for testing, '
                             'but can be set to tr  ue to do the same for training. useful in job scheduler environment,'
                             'where source code might change before the job actually runs.')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='number of using gpu.')
    parser.add_argument('--debug', '-d', action='store_false', default=False)

    parser.add_argument('--fold', type=str, default=0, )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.mode == 'train' or args.mode == 'train_test':
        cf = util.prep_exp(args.exp_source, args.exp_result, args.use_stored_settings, is_train=True)
        if args.debug:
            cf.n_workers = 0
            cf.batch_size = 1
        cf.fold = int(args.fold)
        if args.resume_to_checkpoint is True:
            cf.resume_to_checkpoint = os.path.join(cf.select_model_dir, 'last_state.pth')
        else:
            cf.resume_to_checkpoint = args.resume_to_checkpoint

      

        logger = get_logger(cf.exp_result)
        train(logger, cf)  # , model, dataset
        cf.resume_to_checkpoint = False
         
        if args.mode == 'train_test':
            if not os.path.exists(os.path.dirname(cf.result_csv_path)):
                os.makedirs(os.path.dirname(cf.result_csv_path))
            cf.resume_to_checkpoint = 'result/thyroid_common_npz/epoch_model/model_0.pth'  
            t_test(logger, cf)
        logger.info('OVER!')

    elif args.mode == 'test':
        cf = util.prep_exp(args.exp_source, args.exp_result, args.use_stored_settings, is_train=False)
        cf.fold = int(args.fold)

        logger = get_logger(cf.exp_result)
        cf.resume_to_checkpoint ='result/thyroid_common_npz/best_model.pth'

        t_test(logger, cf)


if __name__ == '__main__':
    main()

