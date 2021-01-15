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


parser = argparse.ArgumentParser(description='Cache IntPhys embeddings')
parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--embedding-model', default='in', type=str, choices=['say', 'in', 'rand'], help='which model to use for embedding')
parser.add_argument('--dynamics-data', default='a', type=str, choices=['s', 'a', 'y', 'intphys'], help='which data to use for training dynamics model')
parser.add_argument('--data-dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
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

    if args.embedding_model == 'say':
        emb_model = models.resnext50_32x4d(pretrained=False)
        emb_model.fc = torch.nn.Linear(in_features=2048, out_features=6269, bias=True)
        emb_model = torch.nn.DataParallel(emb_model).cuda()

        model_path = '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/self_supervised_models/TC-SAY-resnext.tar'

        if os.path.isfile(model_path):
            print("=> loading the model at: '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            emb_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
        emb_model.module.fc = torch.nn.Identity()  # dummy layer

    elif args.embedding_model == 'in':
        print("=> loading the ImageNet pre-trained model")
        emb_model = models.resnext50_32x4d(pretrained=True)
        emb_model.fc = torch.nn.Identity()  # dummy layer
        emb_model = torch.nn.DataParallel(emb_model).cuda()
    elif args.embedding_model == 'rand':
        print("=> loading the untrained model")
        emb_model = models.resnext50_32x4d(pretrained=False)
        emb_model.fc = torch.nn.Identity()  # dummy layer
        emb_model = torch.nn.DataParallel(emb_model).cuda()

    projection = np.load('../caches_saycam/' + args.embedding_model + '_' + args.dynamics_data + '_15fps_1024' + '.npz')['W']
    projection = torch.from_numpy(projection).float().transpose(0, 1).cuda()

    dyn_model = TransformerModel(1024, 4, 1024, 4, 0.05).cuda()
    checkpoint = torch.load(args.embedding_model + '_' + args.dynamics_data + '_model.tar')
    dyn_model.load_state_dict(checkpoint['model_state_dict'])

    dir_list = os.listdir(args.data_dir)
    dir_list.sort()

    scores = {}

    for d in dir_list:

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, d),
            transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None
        )

        embeddings = evaluate(train_loader, emb_model, args)
        embeddings = torch.matmul(embeddings, projection)
        embeddings = torch.unsqueeze(embeddings, 1)

        dyn_model.eval()
    
        with torch.no_grad():
            output = dyn_model(embeddings)
            implausibility = torch.mean(torch.abs(embeddings[1:, :, :] - output[:-1, :, :]), (0, 2)).cpu().numpy()
            print('Clip directory:', os.path.join(args.data_dir, d), 'Implausibility: {:5.8f}'.format(implausibility[0]))

        scores[d] = float(implausibility[0])

    with open(args.embedding_model + '_' + args.dynamics_data + '_' + 'adept_scores.json', 'w') as json_file:
        json.dump(scores, json_file)

    return


def evaluate(data_loader, model, args):

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # note we loop only once
        for _, (images, _) in enumerate(data_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            embeddings = model(images)

    return embeddings


if __name__ == '__main__':
    main()