#-*- coding:UTF-8 -*-
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, process_dataset
from models import LAGNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora',
                    help='dataset cora, citeseer, pubmed, coauthor-cs, coauthor-phy,  amazon-com.')
parser.add_argument('--middle_layer_num', type=int, default=30,
                    help='Number of hidden units.')# 0 2 4 8 14 30
parser.add_argument('--value_layer_dropout', type=float, default=0.05,
                    help='probility of layer_dropout(keep).') # 0 0.5 0.3 0.2 0.1 0.05
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--temperature', type=float, default=0.05,
                    help='temperature in gumbel softmax.')
parser.add_argument('--device', type=int, default=0,
                    help='GPU device.')
parser.add_argument('--epochs', type=int, default=4000,
                    help='Number of epochs to train.')
parser.add_argument('--stage1_epoch', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--weight_choose_share', default=True,
                    help='share choose weight between layer')
parser.add_argument('--choose_weight_type', type=int, default=1,
                    help='0 for input, 1 for input and output, 2 for resnet, 3 for random dropout layer.')
parser.add_argument('--choose_weight_layernum', type=int, default=2,
                    help='the number of weight layer.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.75, 
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--linear_decay', type=int, default=1,
                    help='whether probability decays linearly with the number of layers or not')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print("***********************Hyper-parameter**********************")
print(args)
print("torch.cuda.is_available:",torch.cuda.is_available())

if torch.cuda.is_available() and args.device >= 0:
    dev = torch.device('cuda:%d' % args.device)
    gpu = lambda x: x.to(dev)

np.random.seed()
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = process_dataset(dataset=args.dataset,device = args.device)

model = LAGNN(nfeat=features.shape[1],
            nhid=args.hidden,   # 16
            nmiddle_num=args.middle_layer_num,  # 2
            nclass=labels.max().item() + 1, # 3
            args_parameter=args)

if args.cuda:
    model = model.cuda()
    gpu(model)
    features.cuda()
    gpu(features)
    adj.cuda()
    gpu(adj)

    labels.cuda()
    idx_train.cuda()
    idx_val.cuda()
    idx_test.cuda()
    gpu(labels)
    gpu(idx_train)
    gpu(idx_val)
    gpu(idx_test)

optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

layer_dropout = []
if args.linear_decay != 0:
    for i in range(args.middle_layer_num):
        layer_dropout.append(1 - (i + 1) / (args.middle_layer_num) * (1 - args.value_layer_dropout))

layer_dropout = [args.value_layer_dropout] * args.middle_layer_num
def random_layer():
    layer_choice = [0.0]*args.middle_layer_num
    for i in range(args.middle_layer_num):
        prob = random.random()
        if prob <= layer_dropout[i]:
            layer_choice[i] = 1.0
    return layer_choice

stage1_flag = True
max_acc_train = 0.0
min_loss_train = 10000
max_acc_val = 0.0
output_test_acc = 0.0
min_loss_val = 10000
min_choice = None
max_choice = None
mean_choice = None

def train(epoch):
    global stage1_flag
    global max_acc_train
    global min_loss_train
    global max_acc_val
    global min_loss_val
    global output_test_acc
    global min_choice
    global max_choice
    global mean_choice
    if(stage1_flag == True and (epoch + 1) == args.stage1_epoch or args.stage1_epoch <= 0):
        stage1_flag = False

    t = time.time()
    model.train()
    optimizer.zero_grad()

    layer_choice = random_layer()
    output, node_choice = model(features, adj, layer_choice, stage1_flag)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output, node_choice = model(features, adj, layer_dropout, stage1_flag)
        if stage1_flag == False:
            choice = node_choice[0].clone()
            for i in range(1, len(node_choice)):
                choice = torch.concat([choice, node_choice[i]], dim=1)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

    if loss_val < min_loss_val:
        min_loss_val = loss_val
        output_test_acc = acc_test
        max_acc_val = acc_val
        if stage1_flag == False:
            max_choice = choice.sum(1).max().item()
            min_choice = choice.sum(1).min().item()
            mean_choice = choice.sum(1).mean().item()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.3f}'.format(loss_train.item()),
          'acc_train: {:.3f}'.format(acc_train.item()),
          'loss_val: {:.3f}'.format(loss_val.item()),
          'acc_val: {:.3f}'.format(acc_val.item()),
          'loss_test: {:.3f}'.format(loss_test.item()),
          'acc_test: {:.3f}'.format(acc_test.item())
          )

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
    if((epoch + 1) == args.stage1_epoch):
        print("Stage1 Optimization Finished!")
        print('acc_val: {:.3f}'.format(max_acc_val.item()), 'acc_test: {:.3f}'.format(output_test_acc.item()))
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('acc_val: {:.3f}'.format(max_acc_val.item()),', acc_test: {:.3f}'.format(output_test_acc.item()))
print('mean_choice_num: {:.2f}'.format(mean_choice),
      ', max_choice_num: {:.2f}'.format(max_choice),
      ', min_choice_num: {:.2f}'.format(min_choice))
print("***********************Hyper-parameter**********************")
print(args)
