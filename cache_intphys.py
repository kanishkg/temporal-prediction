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
parser.add_argument('--model', default='say', type=str, choices=['say', 'in', 'rand'], help='which model to use for caching')
parser.add_argument('--data-dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
parser.add_argument('--data', default='train', type=str, choices=['train', 'dev_O1', 'dev_O2', 'dev_O3', 'test_O1', 'test_O2', 'test_O3'], help='which subset of intphys')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='launch N processes per node, which has N GPUs.')
parser.add_argument('--pca', default=False, action='store_true', help='whether to compress embeddings?')


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

    if args.model == 'say':
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=6269, bias=True)
        model = torch.nn.DataParallel(model).cuda()

        model_path = '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/self_supervised_models/TC-SAY-resnext.tar'

        if os.path.isfile(model_path):
            print("=> loading model: '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        model.module.fc = torch.nn.Identity()  # dummy layer
    elif args.model == 'in':
        print('Loading the ImageNet pre-trained model')
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Identity()  # dummy layer
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'rand':
        print('Loading the untrained model')
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Identity()  # dummy layer
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.data == 'dev_O1':
        data_path = os.path.join(args.data_dir, 'dev/O1')
    elif args.data == 'dev_O2':
        data_path = os.path.join(args.data_dir, 'dev/O2')
    elif args.data == 'dev_O3':
        data_path = os.path.join(args.data_dir, 'dev/O3')
    elif args.data == 'test_O1':
        data_path = os.path.join(args.data_dir, 'test/O1')
    elif args.data == 'test_O2':
        data_path = os.path.join(args.data_dir, 'test/O2')
    elif args.data == 'test_O3':
        data_path = os.path.join(args.data_dir, 'test/O3')
    elif args.data == 'train':
        data_path = os.path.join(args.data_dir, 'train')

    train_dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    embeddings, components = evaluate(train_loader, model, args)
    
    if args.data == 'dev_O1': 
        with open(os.path.join(args.data_dir, 'O1_dev_labels.json')) as jf_1:
            label_dict_1 = json.load(jf_1)
        labels_1 = np.array(list(label_dict_1.values()))
        np.savez('intphys_dev_O1_' + args.model, x=embeddings, W=components, y=labels_1)
    elif args.data == 'dev_O2':
        with open(os.path.join(args.data_dir, 'O2_dev_labels.json')) as jf_2:
            label_dict_2 = json.load(jf_2)
        labels_2 = np.array(list(label_dict_2.values()))
        np.savez('intphys_dev_O2_' + args.model, x=embeddings, W=components, y=labels_2)
    elif args.data == 'dev_O3':
        with open(os.path.join(args.data_dir, 'O3_dev_labels.json')) as jf_3:
            label_dict_3 = json.load(jf_3)
        labels_3 = np.array(list(label_dict_3.values()))
        np.savez('intphys_dev_O3_' + args.model, x=embeddings, W=components, y=labels_3)
    elif args.data == 'test_O1': 
        with open(os.path.join(args.data_dir, 'O1_test_labels.json')) as jf_1:
            label_dict_1 = json.load(jf_1)
        labels_1 = np.array(list(label_dict_1.values()))
        np.savez('intphys_test_O1_' + args.model, x=embeddings, W=components, y=labels_1)
    elif args.data == 'test_O2':
        with open(os.path.join(args.data_dir, 'O2_test_labels.json')) as jf_2:
            label_dict_2 = json.load(jf_2)
        labels_2 = np.array(list(label_dict_2.values()))
        np.savez('intphys_test_O2_' + args.model, x=embeddings, W=components, y=labels_2)
    elif args.data == 'test_O3':
        with open(os.path.join(args.data_dir, 'O3_test_labels.json')) as jf_3:
            label_dict_3 = json.load(jf_3)
        labels_3 = np.array(list(label_dict_3.values()))
        np.savez('intphys_test_O3_' + args.model, x=embeddings, W=components, y=labels_3)    
    elif args.data == 'train':
        np.savez('intphys_train_' + args.model, x=embeddings, W=components)

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