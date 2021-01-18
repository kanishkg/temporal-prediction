import argparse
import os
import random
import shutil
import time
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model import TransformerModel
import sklearn.metrics as metrics


parser = argparse.ArgumentParser(description='Cache IntPhys embeddings')
parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--embedding-model', default='in', type=str, choices=['say', 'in', 'rand'], help='which model to use for embedding')
parser.add_argument('--dynamics-data', default='a', type=str, choices=['s', 'a', 'y', 'intphys'], help='which data to use for training dynamics model')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='launch N processes per node, which has N GPUs.')


def main():
    args = parser.parse_args()
    print(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    cudnn.benchmark = True

    projection = np.load('../caches_saycam/' + args.embedding_model + '_' + args.dynamics_data + '_15fps_1024' + '.npz')['W']

    dyn_model = TransformerModel(1024, 4, 1024, 4, 0.1).cuda()
    checkpoint = torch.load(args.embedding_model + '_' + args.dynamics_data + '_model.tar')
    dyn_model.load_state_dict(checkpoint['model_state_dict'])

    intphys_testfile_o1 = 'intphys_test_O1_' + args.embedding_model + '.npz'
    intphys_testfile_o2 = 'intphys_test_O2_' + args.embedding_model + '.npz'
    intphys_testfile_o3 = 'intphys_test_O3_' + args.embedding_model + '.npz'

    intphys_x_o1, intphys_y_o1 = load_intphys(intphys_testfile_o1, projection, 1)
    intphys_x_o2, intphys_y_o2 = load_intphys(intphys_testfile_o2, projection, 0)
    intphys_x_o3, intphys_y_o3 = load_intphys(intphys_testfile_o3, projection, 0)

    o1, y1 = evaluate_intphys(dyn_model, intphys_x_o1, intphys_y_o1)
    o2, y2 = evaluate_intphys(dyn_model, intphys_x_o2, intphys_y_o2)
    o3, y3 = evaluate_intphys(dyn_model, intphys_x_o3, intphys_y_o3)

    return


def load_intphys(filename, projection, ctrl):
    intphys = np.load('../intphys/' + filename)
    intphys_x = intphys['x']
    intphys_x = np.dot(intphys_x, projection.T)
    intphys_y = intphys['y']
    intphys_x = torch.from_numpy(intphys_x).float().transpose(0, 1).cuda()

    # two of the clips in O1 contain all possible clips, the following is a quick fix for that 
    if ctrl:
        intphys_y[911*4] = 0
        intphys_y[911*4 + 1] = 0
        intphys_y[994*4] = 0
        intphys_y[994*4 + 1] = 0

    return intphys_x, intphys_y


def evaluate_intphys(model, intphys_x, intphys_y):

    model.eval()

    with torch.no_grad():
        for i in range(36):
            batch_output = model(intphys_x[:, 120*i:120*(i+1), :])
            if i == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), 1)

        assert output.size(1) == 4320        

        implausibility_signal = torch.mean(torch.abs(intphys_x[1:, :, :] - output[:-1, :, :]), 2)

        intphys_loss = torch.mean(torch.abs(intphys_x[1:, :, :] - output[:-1, :, :]), (0, 2))

        intphys_loss = intphys_loss.cpu().numpy()
        impossibles = np.sum(intphys_loss[intphys_y==0].reshape(1080, 2), axis=1)
        possibles = np.sum(intphys_loss[intphys_y==1].reshape(1080, 2), axis=1)

        correct_frac_rel = (np.sum(possibles <= impossibles) + np.sum(possibles < impossibles)) / 2160.0
        correct_frac_abs = metrics.roc_auc_score(1 - intphys_y, intphys_loss)

        print('IntPhys correct frac (rel) is ', correct_frac_rel, 'IntPhys correct frac (abs) is ', correct_frac_abs)
    
    return implausibility_signal.cpu().numpy(), intphys_y


if __name__ == '__main__':
    main()