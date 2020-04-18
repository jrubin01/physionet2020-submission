#!/usr/bin/env python

import numpy as np
import os
import collections
import pickle

import torch
import torch.nn as nn

from model import TCN


flag_useCuda = True
device = torch.device('cpu')
if flag_useCuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda')


def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)

    # Use your classifier here to obtain a label and score for each class. 
    # normalize the input ECG matrix
    x_in = torch.from_numpy((data.T - model['ecgMean'])/model['ecgStd']).float()
    if flag_useCuda:
        x_in = x_in.cuda()
    
    # order of class names in the trained model
    clsNames = ['AF','I-AVB','LBBB','Normal','RBBB','PAC','PVC','STD','STE']
    
    with torch.no_grad():
        pred_prob = model['Joint'].forward(x_in)[-1].detach().cpu().numpy()
        for modelName in ['AF', 'I-AVB', 'PVC']:
            pred_prob[clsNames.index(modelName)] = model[modelName].forward(\
                x_in)[-1].detach().cpu().numpy()[0]
    
    # re-order the class prediction probability in sorted order
    sorted_index = sorted(range(len(clsNames)), key=lambda k: clsNames[k])
    pred_prob_reorder = pred_prob[sorted_index]
    pred_label_reorder = (pred_prob_reorder>=model['globalOptimalThreshold']).astype(int)

    return pred_label_reorder, pred_prob_reorder


def load_12ECG_model():
    # load models
    db_in = "trained_model/"
    models = collections.OrderedDict()
    n_feature, kernel_size, dropout, num_channels = 12, 16, 0.05, [25] * 11

    for modelName in ['Joint', 'AF', 'I-AVB', 'PVC']:
        # load model
        n_class = 1
        if modelName == 'Joint':
            n_class = 9
        models[modelName] = TCN(n_feature, n_class, num_channels, \
            kernel_size=kernel_size, dropout=dropout)
        if flag_useCuda:
            models[modelName].load_state_dict(torch.load(db_in+modelName+\
                "/trained_model.pt", map_location="cuda:0"))
        else:
            models[modelName].load_state_dict(torch.load(db_in+modelName+\
                "/trained_model.pt", map_location= lambda storage, loc: storage))
        # switch to the evaluation mode
        models[modelName] = models[modelName].eval()
        # move model to GPU if use cuda
        if flag_useCuda:
            models[modelName] = models[modelName].cuda()
    
    # load normalization constant
    with open(db_in+"normalization_constants.pkl", "rb") as pkl_file:
        normalization_constants = pickle.load(pkl_file)
        for key,val in normalization_constants.items():
            models[key] = val

    # load threshold
    with open(db_in+"globalOptimalThreshold.pkl", "rb") as pkl_file:
        models['globalOptimalThreshold'] = pickle.load(pkl_file)

    return models


