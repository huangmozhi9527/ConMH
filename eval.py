import io
import os
import time
import h5py
import numpy as np
import scipy.io as sio
import logging
import argparse

import torch
from torch.autograd import Variable

from configs import Config
from model import get_model
from dataset import get_eval_data
from inference import get_inference
from utils import mAP, set_log, set_seed
import pdb


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


class Array():
    def __init__(self):
        pass

    def setmatrcs(self, matrics):
        self.matrics = matrics

    def concate_v(self, matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics


def evaluate(cfg, model, num_sample, logger):
    model.eval()

    logger.info('eval data number: {}'.format(num_sample))

    logger.info('loading eval data ......') 
    hashcode = np.zeros((num_sample, cfg.nbits), dtype = np.float32)
    label_array = Array()
    hashcode_array = Array()
    rem = num_sample % cfg.test_batch_size
    eval_loader = get_eval_data(cfg)
    eval_loader.dataset.set_mode('test')
    for i, one_label_path in enumerate(cfg.label_path):
        if i == 0:
            if cfg.dataset == 'activitynet':
                labels = sio.loadmat(one_label_path)['re_label']
            else:
                labels = sio.loadmat(one_label_path)['labels']
        else:
            labels = np.concatenate((labels, sio.loadmat(one_label_path)['labels']), axis=0)
    label_array.setmatrcs(labels)
    
    batch_num = len(eval_loader)
    time0 = time.time()
    for i, data in enumerate(eval_loader):
        BinaryCode = get_inference(cfg, data, model)

        if i == batch_num - 1:
            hashcode[i*cfg.test_batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
        else:
            hashcode[i*cfg.test_batch_size:(i+1)*cfg.test_batch_size,:] = BinaryCode.data.cpu().numpy()

    test_hashcode = np.matrix(hashcode)

    if cfg.dataset == 'fcv':
        time1 = time.time()
        logger.info('retrieval costs: {}'.format(time1 - time0))
        Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + cfg.nbits)
        time2 = time.time()
        logger.info('hamming distance computation costs: {}'.format(time2 - time1))
        HammingRank = np.argsort(Hamming_distance, axis=0)
        time3 = time.time()
        logger.info('hamming ranking costs: {}'.format(time3 - time2))

        labels = label_array.getmatrics()
        logger.info('labels shape: {}'.format(labels.shape))
        sim_labels = np.dot(labels, labels.transpose())
        time6 = time.time()
        logger.info('similarity labels generation costs: {}'.format(time6 - time3))
    elif cfg.dataset == 'yfcc' or cfg.dataset == 'activitynet':
        logger.info('loading query data ......') 
        query_hashcode = np.zeros((cfg.query_num_sample, cfg.nbits), dtype = np.float32)
        query_label_array = Array()
        query_hashcode_array = Array()
        query_rem = cfg.query_num_sample % cfg.test_batch_size
        eval_loader.dataset.set_mode('query')
        for i, one_label_path in enumerate(cfg.query_label_path):
            if i == 0:
                if cfg.dataset == 'activitynet':
                    query_labels = sio.loadmat(one_label_path)['q_label']
                else:
                    query_labels = sio.loadmat(one_label_path)['labels']
            else:
                query_labels = np.concatenate((query_labels, sio.loadmat(one_label_path)['labels']), axis=0)
        query_label_array.setmatrcs(query_labels)
        batch_num = len(eval_loader)
        for i, data in enumerate(eval_loader):
            query_BinaryCode = get_inference(cfg, data, model)
            if i == batch_num - 1:
                query_hashcode[i*cfg.test_batch_size:,:] = query_BinaryCode[:query_rem,:].data.cpu().numpy()
            else:
                query_hashcode[i*cfg.test_batch_size:(i+1)*cfg.test_batch_size,:] = \
                                                    query_BinaryCode.data.cpu().numpy()

        query_hashcode = np.matrix(query_hashcode)
        time1 = time.time()
        logger.info('retrieval costs: {}'.format(time1 - time0))
        Hamming_distance = 0.5 * (-np.dot(test_hashcode, query_hashcode.transpose()) + cfg.nbits)

        time2 = time.time()
        logger.info('hamming distance computation costs: {}'.format(time2 - time1))
        HammingRank = np.argsort(Hamming_distance, axis=0)
        time3 = time.time()
        logger.info('hamming ranking costs: {}'.format(time3 - time2))
        
        query_labels = query_label_array.getmatrics()
        labels = label_array.getmatrics()
        logger.info('labels shape: {} and {}'.format(query_labels.shape, labels.shape))
        sim_labels = np.dot(labels, query_labels.transpose())
        time6 = time.time()
        logger.info('similarity labels generation costs: {}'.format(time6 - time3))
    
    maps = []
    map_list = [5,10,20,40,60,80,100]
    for i in map_list:
        map, _, _ = mAP(sim_labels, HammingRank, i)
        maps.append(map)
        logger.info('topK: {}:, map: {}'.format(i, map))
    time7 = time.time()



if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set logging
    logger = set_log(cfg, 'map.txt')
    logger.info('Self Supervised Video Hashing Evaluation: {}'.format(cfg.model_name))

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

    checkpoint = torch.load(cfg.file_path + '/{}_{}.pth'.format(cfg.dataset, cfg.nbits))
    model.load_state_dict(checkpoint['model_state_dict'])

    num_sample = cfg.test_num_sample
    evaluate(cfg, model, num_sample, logger)
    