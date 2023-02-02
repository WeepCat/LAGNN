#-*- coding:UTF-8 -*-
import torch
import math
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, MDCG

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class LAGNN(nn.Module):
    def __init__(self, nfeat, nhid, nmiddle_num, nclass, args_parameter):
        super(LAGNN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)

        self.weight_choose = None
        if args_parameter.weight_choose_share:
            dim = nhid
            if args_parameter.choose_weight_type == 1:
                dim = nhid * 2
            self.weight_choose = nn.ParameterList()
            self.choose_weight_layernum = args_parameter.choose_weight_layernum
            for i in range(self.choose_weight_layernum - 1):
                self.weight_choose.append(Parameter(torch.FloatTensor(dim, dim)))
            self.weight_choose.append(Parameter(torch.FloatTensor(dim, 2)))

            self.reset_parameters()
        self.gc_middle = nn.ModuleList()
        for i in range(nmiddle_num):
            self.gc_middle.append(MDCG(nhid, nhid, args_parameter, self.weight_choose))
        self.gcl = GraphConvolution(nhid, nclass)
        self.dropout = args_parameter.dropout
        self.device = args_parameter.device

    def reset_parameters(self):
        for i in range(self.choose_weight_layernum):
            stdv = 1. / math.sqrt(self.weight_choose[i].size(1))
            self.weight_choose[i].data.uniform_(-stdv, stdv)

    def forward(self, x, adj, layer_dropout, stage1_flag):
        activations = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        activations.append(x)
        node_lastlayer = [torch.ones(x.shape[0]).view([-1, 1])]

        for i in range(len(self.gc_middle)):
            x, node_choose = self.gc_middle[i](x, adj, self.dropout, layer_dropout[i], node_lastlayer[-1], stage1_flag)
            activations.append(x)
            node_lastlayer.append(node_choose)

        if not stage1_flag and len(self.gc_middle) > 0:
            node_lastlayer.append(torch.zeros(x.shape[0]).view([-1, 1]))

            if torch.cuda.is_available() and self.device >= 0:
                dev = torch.device('cuda:%d' % self.device)
                gpu = lambda x: x.to(dev)
            for i in range(len(node_lastlayer)):
                node_lastlayer[i] = gpu(node_lastlayer[i])
            for i in range(len(activations)):
                activations[i] = gpu(activations[i])

            x = (node_lastlayer[0] - node_lastlayer[1]) * activations[0]
            for i in range(1, len(self.gc_middle) + 1):
                x = x + ((node_lastlayer[i] - node_lastlayer[i + 1]) * activations[i])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcl(x, adj)

        return F.log_softmax(x, dim=1), node_lastlayer