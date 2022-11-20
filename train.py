import os
import argparse
import numpy as np
import scipy.io as sio
import pickle
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable

from configs import Config
from model import get_model
from dataset import get_train_data, get_eval_data
from optim import get_opt_schedule, set_lr
from Loss import get_loss
from utils import set_log, set_seed

from eval import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='ssvh')
    parser.add_argument('--config', default='configs/conmh_fcv.py', type = str,
        help='config file path'
    )
    parser.add_argument('--gpu', default = '0', type = str,
        help = 'specify gpu device'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if not os.path.exists(cfg.file_path):
        os.makedirs(cfg.file_path)
    
    # set logging
    logger = set_log(cfg, 'log.txt')
    logger.info('Self Supervised Video Hashing Training: {}'.format(cfg.model_name))

    # set seed
    set_seed(cfg)
    logger.info('set seed: {}'.format(cfg.seed))

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # hyper parameter
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    logger.info('PARAMETER ......')
    logger.info(cfg)

    logger.info('loading model ......') 
    model = get_model(cfg).cuda()

    if len(device_ids) > 1:
        model = nn.DataParallel(model)

    logger.info('loading train data ......')    
    train_loader = get_train_data(cfg)
    total_len = len(train_loader)

    epoch = 0

    # optimizer and schedule
    opt_schedule = get_opt_schedule(cfg, model)

    if cfg.use_checkpoint is not None:
        checkpoint = torch.load(cfg.use_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_schedule._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        opt_schedule._schedule.last_epoch = checkpoint['epoch']
    

    while True:
        if cfg.dataset == 'fcv':
            if epoch % 50 == 0 and epoch > 0:
                evaluate(cfg, model, cfg.test_num_sample, logger)
        elif cfg.dataset == 'activitynet':
            if epoch % 20 == 0:
                evaluate(cfg, model, cfg.test_num_sample, logger)
        elif cfg.dataset == 'yfcc':
            if epoch == 40:
                evaluate(cfg, model, cfg.test_num_sample, logger)

        logger.info('begin training stage: [{}/{}]'.format(epoch+1, cfg.num_epochs))  
        model.train()

        for i, data in enumerate(train_loader, start=1):
            opt_schedule.zero_grad()

            loss = get_loss(cfg, data, model, epoch, i, total_len, logger)

            loss.backward()
            opt_schedule._optimizer_step()

        opt_schedule._schedule_step()
        logger.info('now the learning rate is: {}'.format(opt_schedule.lr()))

        if epoch == cfg.num_epochs - 6:
            save_file = cfg.file_path + '/{}_{}.pth'.format(cfg.dataset, cfg.nbits)
            torch.save({
                'model_state_dict': model.state_dict()
            }, save_file)

        save_file = cfg.file_path + '/model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt_schedule._optimizer.state_dict()
        }, save_file)
        
        epoch += 1
        if epoch >= cfg.num_epochs:
            break

if __name__ == '__main__':
    main()
