#-*- coding:UTF-8 -*-
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MDCG(Module):
    def __init__(self, in_features, out_features, args_parameter, weight_choose, bias=True):
        super(MDCG, self).__init__()
        # parameter for gumbel_softmax
        self.temperature = args_parameter.temperature
        # considering the input and output
        self.choose_weight_type = args_parameter.choose_weight_type
        # the num of choice layer
        self.choose_weight_layernum = args_parameter.choose_weight_layernum
        self.device = args_parameter.device
        # whether sharing the parameters for choice layer or not
        self.weight_choose_share = args_parameter.weight_choose_share
        if self.weight_choose_share:
            self.weight_choose = weight_choose
        else:
            dim = in_features
            if self.choose_weight_type == 1:    # concat the input and output
                dim = in_features * 2

            self.weight_choose = nn.ParameterList()
            for i in range(args_parameter.choose_weight_layernum - 1):
                self.weight_choose.append(Parameter(torch.FloatTensor(dim, dim)))
            self.weight_choose.append(Parameter(torch.FloatTensor(dim, 2)))

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Initializing the parameter
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if not self.weight_choose_share:
            for i in range(self.choose_weight_layernum):
                stdv = 1. / math.sqrt(self.weight_choose[i].size(1))
                self.weight_choose[i].data.uniform_(-stdv, stdv)


    def forward(self, input, adj, cell_dropout, layer_dropout, node_lastlayer, stage1_flag):
        x = F.dropout(input, cell_dropout, training=self.training)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = F.relu(output)

        node_choose = None
        # stage1:
        if stage1_flag:
            output = input + output
        # stage2:
        else:
            choose_input = input
            if self.choose_weight_type == 1: # concat the input and output
                choose_input = torch.cat((input, input + output), 1)

            node_choose = torch.mm(choose_input, self.weight_choose[0])

            for i in range(1, self.choose_weight_layernum):
                choose_input = F.relu(node_choose)
                node_choose = torch.mm(choose_input, self.weight_choose[i])

            # whether choice the layer or not
            node_choose = F.gumbel_softmax(node_choose, tau=self.temperature, hard=True, dim=1)[:, 0].view(-1, 1)
            if torch.cuda.is_available() and self.device >= 0:
                dev = torch.device('cuda:%d' % self.device)
                gpu = lambda x: x.to(dev)
            node_lastlayer = gpu(node_lastlayer)
            node_choose = node_lastlayer * node_choose.view([-1, 1])
            output = input + layer_dropout * output
        return output, node_choose

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'