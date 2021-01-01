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
parser.add_argument('--data', type=str, default='intphys', choices=['TC_SAY_S1_15fps_2048_IN', 'TC_SAY_A_15fps_2048_IN', 'TC_SAY_Y_15fps_2048_IN', 'intphys'], help='cached frame embeddings')
parser.add_argument('--model', type=str, default='Transformer', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--nhid', type=int, default=512, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4, help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=16000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--nhead', type=int, default=4, help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true', help='verify the code and the model')
parser.add_argument('--val-frac', type=float, default=0.01, help='fraction of whole data reserved for validation')

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
    x_W = np.load('../intphys/intphys_train.npz')
    whole_data = x_W['x']
else:
    x_W = np.load('../caches_IN/' + args.data + '.npz')  # T x emsize
    whole_data = x_W['x']
    whole_data = whole_data[:2400000, :]
    whole_data = whole_data.reshape(24000, 100, 2048)

num_data = whole_data.shape[0]
val_size = int(args.val_frac * num_data)

train_data = whole_data[val_size:]
val_data = whole_data[:val_size]

num_iters_per_epoch = train_data.shape[0] // args.batch_size

print('Loaded data')
print('Data size: ', whole_data.shape)
print('Train size: ', train_data.shape)
print('Val. size: ', val_data.shape)
print('Number of training iterations per epoch', num_iters_per_epoch)

# As a baseline, compute the loss for predicting the previous embedding
prev_loss = np.mean(abs(val_data[:, 1:, :] - val_data[:, :-1, :]))
print('Previous embedding loss on val. data is ', prev_loss)

# convert to tensor
train_data = torch.from_numpy(train_data).float().transpose(0, 1).cuda()
val_data = torch.from_numpy(val_data).float().transpose(0, 1).cuda()
emsize = train_data.size(2)

###############################################################################
# Load intphys
###############################################################################

intphys_devfile_o1 = 'intphys_dev_O1.npz'
intphys_devfile_o2 = 'intphys_dev_O2.npz'
intphys_devfile_o3 = 'intphys_dev_O3.npz'

def load_intphys(filename):
    intphys = np.load('../intphys/' + filename)
    intphys_x = intphys['x']
    intphys_x = np.dot(intphys_x, x_W['W'].T)
    intphys_y = intphys['y']
    intphys_x = torch.from_numpy(intphys_x).float().transpose(0, 1).cuda()

    return intphys_x, intphys_y

intphys_x_o1, intphys_y_o1 = load_intphys(intphys_devfile_o1)
intphys_x_o2, intphys_y_o2 = load_intphys(intphys_devfile_o2)
intphys_x_o3, intphys_y_o3 = load_intphys(intphys_devfile_o3)

###############################################################################
# Build the model
###############################################################################

if args.model == 'Transformer':
    model = model.TransformerModel(emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), args.lr)
lr_lambda = lambda epoch: 0.999 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, batch_size):
    batch_idx = np.random.randint(0, source.size(1), size=batch_size)
    data = source[:-1, batch_idx, :]
    target = source[1:, batch_idx, :]
    return data, target

def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    inputs = val_data[:-1, :, :]
    targets = val_data[1:, :, :]
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        if args.model == 'Transformer':
            output = model(inputs)
        else:
            output, hidden = model(inputs, hidden)
            hidden = repackage_hidden(hidden)
        total_loss = criterion(output, targets).item()
    return total_loss

def evaluate_intphys(intphys_x, intphys_y):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    
    with torch.no_grad():
        if args.model == 'Transformer':
            output = model(intphys_x)
        else:
            output, hidden = model(intphys_x, hidden)
            hidden = repackage_hidden(hidden)

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
val_losses = []

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    for i in range(0, num_iters_per_epoch):
        data, targets = get_batch(train_data, args.batch_size)
        model.zero_grad()

        if args.model == 'Transformer':
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            tr_losses.append(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} mini-batches | lr {:02.6f} | ms/batch {:5.2f} | tr loss {:5.4f} '.format(epoch, i, num_iters_per_epoch, scheduler.get_last_lr()[0], elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        scheduler.step()
        o1, y1 = evaluate_intphys(intphys_x_o1, intphys_y_o1)
        o2, y2 = evaluate_intphys(intphys_x_o2, intphys_y_o2)
        o3, y3 = evaluate_intphys(intphys_x_o3, intphys_y_o3)
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        val_losses.append(val_loss)
        np.savez(args.model + '_' + args.data + '_log.npz', o1=o1, o2=o2, o3=o3, y1=y1, y2=y2, y3=y3, tr_losses=np.array(tr_losses), val_losses=np.array(val_losses))
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')