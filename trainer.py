from metrics import get_metrics

import numpy as np
import torch
import torch.nn as nn
import scipy

def train_classifier(device, network, trainset, testset, hyper_train, binenc_loss=True, verbose=True):

    batch_size = hyper_train['batch_size']
    epochs = hyper_train['epochs']
    step_scheduler = hyper_train['step_scheduler']
    lr = hyper_train['lr']
    loss_pen_factor = hyper_train['loss_pen_factor']
    logging_pen = hyper_train['logging_pen']
    logging = hyper_train['logging']

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True )
    
    opt = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_scheduler, gamma=0.5)
    res_list = []
    res_training = {}

    for epoch in range (epochs):
        for x,y in trainloader:
            x=x.to(device)
            y=y.to(device)
            opt.zero_grad()
            y_pred, pen_layer= network(x)
   
            loss_class = nn.CrossEntropyLoss(reduction='mean')(y_pred,y)
            loss = loss_class 
            if binenc_loss:
                loss_pen = torch.exp(nn.functional.mse_loss(pen_layer, torch.zeros(pen_layer.shape).to(device), reduction='mean' ))
                loss = loss + loss_pen_factor*loss_pen
                
            loss.backward()
            opt.step()
        
        if epoch%logging==0 :
            if verbose:
                print('Epoch: ', epoch)
            save_pen = False
            if logging_pen>0 and epoch%logging_pen==0:
                save_pen = True

            res_epoch = get_metrics(device=device, network=network, trainset=trainset, testset=testset, save_pen=save_pen, verbose=verbose)

            res_list.append(res_epoch)

    for key in res_list[0].keys():
        res_training[key] = np.vstack([res_epoch [key] for res_epoch in res_list])
    
    return res_training