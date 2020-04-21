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
    x_in = data.T.astype(float)
    # down-sample from 500 Hz to 250 Hz
    x_in = x_in[np.arange(0,x_in.shape[0],2),:]
    # normalize each channel
    x_in = (x_in-x_in.mean(axis=0))/(x_in.std(axis=0)+1e-6)
    # convert numpy array to pytorch tensor
    x_in = torch.from_numpy(x_in).float()
    if flag_useCuda:
        x_in = x_in.cuda()
    
    # order of class names in the trained model
    clsNames = ['AF','I-AVB','LBBB','Normal','RBBB','PAC','PVC','STD','STE']
    
    # each row is one prediction
    pred_label = np.zeros((13, 9))
    with torch.no_grad():
        for idx_model in range(13):
            if idx_model < 12:
                x_input = x_in[:,idx_model][:,None]
            else:
                x_input = x_in
            pred_prob = model[idx_model].forward(x_input)[-1].\
                detach().cpu().numpy()
            pred_label[idx_model] = (pred_prob>=model[str(idx_model)+'_thd']).\
                astype(float)
    # compute the mean value of the predicted label across 12 channels
    pred_label_mean = pred_label[0:12].mean(axis=0)
    # compute the average over the ensemble of 12 channels and model 12
    pred_label_mean = (pred_label_mean+pred_label[12])/2

    # re-order the class prediction probability in sorted order
    sorted_index = sorted(range(len(clsNames)), key=lambda k: clsNames[k])
    pred_prob_ensemble = pred_label_mean[sorted_index]
    pred_label_ensemble = (pred_prob_ensemble>0.6923).astype(int)
    
    return pred_label_ensemble, pred_prob_ensemble


def load_12ECG_model():
    # load models
    db_in = "./trained_models/"
    models = collections.OrderedDict()
    kernel_size, dropout, num_channels = 32, 0, [25] * 9
    n_class = 9

    for i, channel in enumerate([str(item) for item in range(12)]+\
            [','.join([str(item) for item in range(12)])]):
        # load model
        modelName = "tcn_valFold8_testFold9_target0,1,2,3,4,5,6,7,8_channel"+\
            channel+"_normlocal_downSample1_valMetricauroc_nlayer9_hidden25"+\
            "_ksize32_dropout0_optimizerAdam_batchSize32_epoch100_regl2_"+\
            "decay0.0001_lr0.0001_clip0.2_seed42/"
        n_feature = len(channel.split(","))
        models[i] = TCN(n_feature, n_class, num_channels, \
            kernel_size=kernel_size, dropout=dropout)
        if flag_useCuda:
            models[i].load_state_dict(torch.load(db_in+modelName+\
                "/trained_model.pt", map_location="cuda:0"))
        else:
            models[i].load_state_dict(torch.load(db_in+modelName+\
                "/trained_model.pt", map_location= lambda storage, loc: storage))
        # switch to the evaluation mode
        models[i] = models[i].eval()

        # move model to GPU if use cuda
        if flag_useCuda:
            models[i] = models[i].cuda()

        # load threshold
        with open(db_in+modelName+"clsOptimalThreshold.pkl", "rb") as pkl_file:
            models[str(i)+'_thd'] = pickle.load(pkl_file)['clsOptimalThreshold']

    return models
