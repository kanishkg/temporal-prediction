# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import sklearn.metrics as metrics

import model

parser = argparse.ArgumentParser(description='SAYCam Temporal Prediction Model')
parser.add_argument('--embedding-model', type=str, default='say', choices=['say', 'in', 'rand'], help='embedding model')
parser.add_argument('--data', type=str, default='a', choices=['s', 'a', 'y', 'intphys'], help='data for training the dynamics model')
parser.add_argument('--nhid', type=int, default=1024, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4, help='number of layers')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--nhead', type=int, default=4, help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true', help='verify the code and the model')

args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

if args.data == 'intphys':
    x_W = np.load('../intphys/intphys_train_' + args.embedding_model + '.npz')
    train_data = x_W['x']
else:
    x_W = np.load('../caches_saycam/' + args.embedding_model + '_' + args.data + '_15fps_1024' + '.npz')  # T x emsize
    train_data = x_W['x']
    trim_end = (train_data.shape[0] // 100) * 100
    train_data = train_data[:trim_end, :]
    train_data = train_data.reshape(-1, 100, 1024)

num_iters_per_epoch = train_data.shape[0] // args.batch_size

print('Loaded data')
print('Training data size: ', train_data.shape)
print('Number of training iterations per epoch', num_iters_per_epoch)

# As a baseline, compute the loss for predicting the previous embedding
prev_loss = np.mean(abs(train_data[:, 1:, :] - train_data[:, :-1, :]))
print('Previous embedding loss on training data is ', prev_loss)

# convert to tensor
train_data = torch.from_numpy(train_data).float().transpose(0, 1).cuda()
emsize = train_data.size(2)

###############################################################################
# Load intphys
###############################################################################

def load_intphys(filename):
    intphys = np.load('../intphys/' + filename)
    intphys_x = intphys['x']
    intphys_x = np.dot(intphys_x, x_W['W'].T)
    intphys_y = intphys['y']
    intphys_x = torch.from_numpy(intphys_x).float().transpose(0, 1).cuda()

    return intphys_x, intphys_y

intphys_devfile_o1 = 'intphys_dev_O1_' + args.embedding_model + '.npz'
intphys_devfile_o2 = 'intphys_dev_O2_' + args.embedding_model + '.npz'
intphys_devfile_o3 = 'intphys_dev_O3_' + args.embedding_model + '.npz'

intphys_x_o1, intphys_y_o1 = load_intphys(intphys_devfile_o1)
intphys_x_o2, intphys_y_o2 = load_intphys(intphys_devfile_o2)
intphys_x_o3, intphys_y_o3 = load_intphys(intphys_devfile_o3)

###############################################################################
# Build the model, loss fn, optimizer
###############################################################################

model = model.TransformerModel(emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), args.lr)
lr_lambda = lambda epoch: 0.999 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

###############################################################################
# Training code
###############################################################################

def get_batch(source, batch_size):
    batch_idx = np.random.randint(0, source.size(1), size=batch_size)
    data = source[:-1, batch_idx, :]
    target = source[1:, batch_idx, :]
    return data, target

def evaluate_intphys(intphys_x, intphys_y):

    model.eval()
    
    with torch.no_grad():
        output = model(intphys_x)

        implausibility_signal = torch.mean(torch.abs(intphys_x[1:, :, :] - output[:-1, :, :]), 2)

        intphys_loss = torch.mean(torch.abs(intphys_x[1:, :, :] - output[:-1, :, :]), (0, 2))

        intphys_loss = intphys_loss.cpu().numpy()
        impossibles = np.sum(intphys_loss[intphys_y==0].reshape(30, 2), axis=1)
        possibles = np.sum(intphys_loss[intphys_y==1].reshape(30, 2), axis=1)

        correct_frac_rel = (np.sum(possibles <= impossibles) + np.sum(possibles < impossibles)) / 60.0
        correct_frac_abs = metrics.roc_auc_score(1 - intphys_y, intphys_loss)

        print('IntPhys correct frac (rel) is ', correct_frac_rel, 'IntPhys correct frac (abs) is ', correct_frac_abs)
    
    return implausibility_signal.cpu().numpy(), intphys_y

tr_losses = [] 

def train():

    model.train()
    total_loss = 0.
    
    for _ in range(0, num_iters_per_epoch):
        data, targets = get_batch(train_data, args.batch_size)
        model.zero_grad()

        output = model(data)

        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if args.dry_run:
            break

    tr_losses.append(total_loss / num_iters_per_epoch)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        scheduler.step()
        o1, y1 = evaluate_intphys(intphys_x_o1, intphys_y_o1)
        o2, y2 = evaluate_intphys(intphys_x_o2, intphys_y_o2)
        o3, y3 = evaluate_intphys(intphys_x_o3, intphys_y_o3)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | training loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time), tr_losses[-1]))
        print('-' * 89)
        np.savez(args.embedding_model + '_' + args.data + '_log.npz', o1=o1, o2=o2, o3=o3, y1=y1, y2=y2, y3=y3, tr_losses=np.array(tr_losses))
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.embedding_model + '_' + args.data + '_model' + '.tar')
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')