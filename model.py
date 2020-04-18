import os
import sys
import math
import collections
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer, required


# choose GPU device
flag_useCuda = True
device = torch.device('cpu')
if flag_useCuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda')

# model: TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        #self.pad1 = nn.ReplicationPad1d(padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        #self.pad2 = nn.ReplicationPad1d(padding)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # nn.init.kaiming_normal_(self.conv1.weight.data)
        # nn.init.kaiming_normal_(self.conv2.weight.data)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            #nn.init.kaiming_normal_(self.downsample.weight.data)
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: seqLen x inputSize
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2)
        output = self.linear(output)
        output = self.sig(output).squeeze(0)
        return output

class FSTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(FSTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        # initialize the selector
        self.weight_fs = nn.Parameter(torch.randn(input_size).to(device)*0.3)
    def forward(self, x):
        # x: seqLen x inputSize
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        x = x * 1./(1+torch.exp(-10*self.weight_fs))
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2)
        output = self.linear(output)
        output = self.sig(output).squeeze(0)[:,0]
        return output

class TCNClassifier(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: seqLen x inputSize
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2)
        output = self.linear(output).squeeze(0)
        output = nn.functional.softmax(output, dim=1)
        return output

class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, \
        dropout, device):
        super(MixtureOfExperts, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.smallTCN = TemporalConvNet(input_size, [100], kernel_size, dropout)
        self.gate = TCN(input_size, output_size, [100], kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        #self.device = device
    def forward(self, x):
        # x: seqLen x inputSize
        x_in = x.unsqueeze(0).transpose(1,2)
        # TCN embedding
        prob_tcn = self.tcn.forward(x_in).transpose(1,2).squeeze(0)
        # smallTCN embedding
        prob_smallTCN = self.smallTCN.forward(x_in).transpose(1,2).squeeze(0)
        # gating function
        flag_useTCN = self.gate.forward(x)
        # embedding as the weighted average, embed: L x dim_hid
        embed = prob_tcn*flag_useTCN[:,None] + prob_smallTCN*(1-flag_useTCN[:,None])
        output = self.sig(self.linear(embed))[:,0]
        return output

class MixtureTCNMLP(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, \
        dropout, device):
        super(MixtureTCNMLP, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.smallTCN = TemporalConvNet(input_size, [100], kernel_size, dropout)
        self.mlp = MLPEncoder(dim_input=input_size, dim_hid=100, num_layer=3)
        self.gate = TCNClassifier(input_size, 3, [100], kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        #self.device = device
    def forward(self, x):
        # x: seqLen x inputSize
        x_in = x.unsqueeze(0).transpose(1,2)
        # TCN embedding
        embed_tcn = self.tcn.forward(x_in).transpose(1,2).squeeze(0)
        # smallTCN embedding
        embed_smallTCN = self.smallTCN.forward(x_in).transpose(1,2).squeeze(0)
        # MLP embedding
        embed_mlp = self.mlp.forward(x)
        # gating function
        gate = self.gate.forward(x)
        # embedding as the weighted average, embed: L x dim_hid
        embed = embed_tcn*gate[:,[0]] + embed_smallTCN*gate[:,[1]] + embed_mlp*\
            gate[:,[2]]
        output = self.sig(self.linear(embed))[:,0]
        return output

class TCNRegression(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNRegression, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: seqLen x inputSize
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2)
        output = self.linear(output).squeeze(0)[:,0]
        return output

# implement the wide TCN taking static features as input
class WideTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, \
        dropout, dim_static):
        # dim_static: dimension of the static variable
        super(WideTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]+dim_static, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, x_static):
        # x: seqLen x inputSize
        # x_static: shape(dim_static, )
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2).squeeze(0)
        # concatenate the TCN representation with the static variable
        output = torch.cat([output, x_static.repeat(output.shape[0],1)], 1)
        output = self.linear(output)
        output = self.sig(output)[:,0]
        return output

class RNN(nn.Module):
    def __init__(self, dim_input=1, dim_hid=100, device=torch.device('cpu'), \
            survivalRNN=False, num_layer=1):
        super(RNN, self).__init__()
        self.lstm = nn.LSTMCell(dim_input, dim_hid)
        self.lstm2 = nn.LSTMCell(dim_hid, dim_hid)
        self.linear = nn.Linear(dim_hid, 1)
        self.dim_input, self.dim_hid = dim_input, dim_hid
        self.device = device
        self.survivalRNN = survivalRNN
        self.num_layer=num_layer

    def forward(self, input):
        """
        input: seqLen x inputSize
        survivalRNN: if survivalRNN=True, then treat the RNN output at each
        step as the hazard function in survival model, then the probability of
        event(having sepsis in 6 hours) is written as 1-\prod_t(1-ht)
        """
        # initialize outputs, hidden state, and cell state
        outputs = []
        h_t = torch.zeros(1, self.dim_hid, device=self.device)
        c_t = torch.zeros(1, self.dim_hid, device=self.device)
        h_t2 = torch.zeros(1, self.dim_hid, device=self.device)
        c_t2 = torch.zeros(1, self.dim_hid, device=self.device)

        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            if self.num_layer == 1:
                output = 1/(1+torch.exp(-self.linear(h_t)))
            elif self.num_layer == 2:
                output = 1/(1+torch.exp(-self.linear(h_t2)))
            outputs.append(output)
        outputs = torch.stack(outputs, 1).squeeze()
        if self.survivalRNN:
            outputs = torch.clamp(1-torch.exp(torch.cumsum(torch.log(
                1+1e-6-outputs),dim=0)), min=0, max=1)
        return outputs

class MLP(nn.Module):
    def __init__(self, dim_input=1, dim_hid=100, n_class=2, num_layer=1, \
            survivalMLP=False):
        super(MLP, self).__init__()
        self.layer = nn.ModuleList()
        self.survivalMLP = survivalMLP
        self.num_layer = num_layer
        # logistic regression
        if num_layer == 0:
            self.layer.append(nn.Linear(dim_input, n_class))
        # MLP
        elif num_layer > 0:
            self.layer.append(nn.Sequential(\
                nn.Linear(dim_input, dim_hid)))
            for i in range(num_layer-1):
                self.layer.append(nn.Sequential(\
                    nn.Linear(dim_hid, dim_hid)))
            self.layer.append(nn.Linear(dim_hid, n_class))
    
    def forward(self, x):
        """
        x: seqLen x inputSize
        """
        for i, layer in enumerate(self.layer):
            x = layer(x)
            if i < self.num_layer:
                x = nn.functional.relu(x)
        outputs = nn.functional.softmax(x, dim=1)[:,1]
        if self.survivalMLP:
            outputs = torch.clamp(1-torch.exp(torch.cumsum(torch.log(\
                    1+1e-6-outputs),dim=0)), min=0, max=1)
        return outputs


class MLPEncoder(nn.Module):
    def __init__(self, dim_input=1, dim_hid=100, num_layer=1):
        super(MLPEncoder, self).__init__()
        self.layer = nn.ModuleList()
        self.num_layer = num_layer
        # logistic regression
        if num_layer == 0:
            self.layer.append(nn.Linear(dim_input, n_class))
        # MLP
        elif num_layer > 0:
            self.layer.append(nn.Sequential(\
                nn.Linear(dim_input, dim_hid)))
            for i in range(num_layer-1):
                self.layer.append(nn.Sequential(\
                    nn.Linear(dim_hid, dim_hid)))
    def forward(self, x):
        """
        x: seqLen x inputSize
        """
        for i, layer in enumerate(self.layer):
            x = layer(x)
            if i < self.num_layer:
                x = nn.functional.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self, dim_input=1, n_class=2, num_layer=1, n_channel=20):
        super(CNN, self).__init__()
        self.layer = nn.ModuleList()
        self.num_layer = num_layer       
        for i in range(num_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = n_channel
            self.layer.append(nn.Sequential(\
                nn.Conv1d(in_channel, n_channel, kernel_size=3, padding=1), \
                nn.BatchNorm1d(n_channel), \
                nn.ReLU()))
        self.linear = nn.Linear(n_channel*dim_input, n_class)

    def forward(self, x):
        """
        x: seqLen x inputSize
        """
        # reshape to: seqLen x 1 x inputSize
        x = x[:,None,:]
        for i, layer in enumerate(self.layer):
            x = layer(x)
        x = self.linear(x.view(x.shape[0], -1))
        outputs = nn.functional.softmax(x, dim=1)[:,1]
        return outputs


class AdamW(Optimizer):
    """Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        #super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.mul_(1 - group['weight_decay']).addcdiv_(-step_size, exp_avg, denom)

        return loss


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
