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


parser = argparse.ArgumentParser(description='Cache IntPhys embeddings')
parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--batch-size', default=100, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--model-path', default='', type=str, metavar='PATH', help='path to model checkpoint (default: none)')
parser.add_argument('--data-dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
parser.add_argument('--n_out', default=20, type=int, help='output dim')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='launch N processes per node, which has N GPUs.')
parser.add_argument('--pca', default=False, action='store_true', help='whether to compress embeddings?')


def main():
    args = parser.parse_args()

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

    if args.model_path:
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)
        model = torch.nn.DataParallel(model).cuda()

        if os.path.isfile(args.model_path):
            print("=> loading model: '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

        model.module.fc = torch.nn.Identity()  # dummy layer
    else:
        # torch_hub_dir = '/misc/vlgscratch4/LakeGroup/emin/robust_vision/pretrained_models'
        # torch.hub.set_dir(torch_hub_dir)

        # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        # print(model)
        # model.fc = torch.nn.Identity()  # dummy layer
        # model = torch.nn.DataParallel(model).cuda()

        model = models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Identity()  # dummy layer
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    savefile_name = 'intphys_train_32'

    # with open(os.path.join('/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15', 'O3_dev_labels.json')) as jf:
    #     label_dict = json.load(jf)
    # labels = np.array(list(label_dict.values()))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        args.data_dir,
        transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    embeddings, components = evaluate(train_loader, model, args)
    
    np.savez(savefile_name, x=embeddings, W=components)

    return


def evaluate(data_loader, model, args):

    embeddings = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            embedding = model(images).cpu().numpy()

            print(i, embedding.shape)

            embeddings.append(np.expand_dims(embedding, 0))


    embeddings = np.concatenate(embeddings)
    print('Embeddings shape:', embeddings.shape)

    if args.pca:
        from sklearn.decomposition import IncrementalPCA

        # do incremental PCA
        ipca = IncrementalPCA(n_components=32)
        embeddings = ipca.fit_transform(np.reshape(embeddings, (-1, 2048)))
        print('Embeddings shape 1:', embeddings.shape)
        embeddings = np.reshape(embeddings, (15000, 100, 32))
        components = ipca.components_
        print('Embeddings shape 2:', embeddings.shape)
        print('Components shape:', components.shape)
        print('Successfully completed the online PCA. Variance explained:', np.sum(ipca.explained_variance_ratio_))
    else:
        components = np.eye(2048)

    return embeddings, components


if __name__ == '__main__':
    main()