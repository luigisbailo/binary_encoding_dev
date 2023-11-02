from nn import Classifier_cnn
from trainer import train_classifier

import torch
import torchvision
import numpy as np
import pickle
import sys
import os 
import shutil
import yaml
import argparse
import gdown
import zipfile
import torchvision.transforms as transforms
import importlib

def parse_config(config_file):
    try:
        with open(config_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            return data
    except FileNotFoundError:
        print(f"File not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    configs =  parse_config(parser.parse_args().config)

    name_dataset = configs['dataset']['name']

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    torch_module= importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torch_module, name_dataset)

    trainset = torch_dataset ("./dataset", train=True, download=True)
    testset = torch_dataset ("./dataset", train=False, download=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.is_available()

    batch_size = 64
    epochs = 1
    samples = 1
    pen_layer_node = 64
    hidden_layer_base = [1024,1024]
    loss_pen_factor = 10
    logging_pen = 5
    step_scheduler = 20

    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results")


    res_loss = {}
    res_list = []
    for i in range(samples):
        print('Sample: ', i)
        
        classifier_cnn_loss = Classifier_cnn(pen_layer_node=pen_layer_node, hidden_layer_node=hidden_layer_base).to(device)
        if (i==0):
            res = train_classifier(
                device, classifier_cnn_loss, trainset, testset, accuracy_target=1, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler, 
                pen_loss = True, save_pen = True, logging_pen=logging_pen, pen_layer_node=pen_layer_node, loss_pen_factor=loss_pen_factor)
        else:
            res = train_classifier(
                device, classifier_cnn_loss, trainset, testset, accuracy_target=1, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler, verbose=False,
                pen_loss = True, pen_layer_node=pen_layer_node, loss_pen_factor=loss_pen_factor)
            
        res_list.append(res)
    for key in res_list[0].keys():
        try:
            res_loss[key] = np.hstack([res[key] for res in res_list ])
        except:
            res_loss[key] = res_list[0][key]

    with open ('results/res_loss.pkl', 'wb') as file:
        pickle.dump(res_loss, file)

    res_noloss = {}
    res_list = []
    for i in range(samples):
        print('Sample: ', i)
        
        classifier_cnn_noloss = Classifier_cnn(pen_layer_node=pen_layer_node, hidden_layer_node=hidden_layer_base).to(device)
        if i==0:
            res = train_classifier(
                device, classifier_cnn_noloss, trainset, testset, accuracy_target=1, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler, 
                pen_loss=False, save_pen=True, logging_pen=logging_pen, pen_layer_node=pen_layer_node, loss_pen_factor=loss_pen_factor)  
        else:
            res = train_classifier(
                device, classifier_cnn_noloss, trainset, testset, accuracy_target=1, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler, verbose=False,
                pen_loss=False, pen_layer_node=pen_layer_node, loss_pen_factor=loss_pen_factor)  
        
        res_list.append(res)
    for key in res_list[0].keys():
        try:
            res_noloss[key] = np.hstack([res[key] for res in res_list ])
        except:
            res_noloss[key] = res_list[0][key]
    with open ('results/res_noloss.pkl', 'wb') as file:
        pickle.dump(res_noloss, file)

    res_0 = {}
    res_list = []
    for i in range(samples):
        print('Sample: ', i)
        
        classifier_cnn_0 = Classifier_cnn(pen_layer_node=None, hidden_layer_node=hidden_layer_base).to(device)
        res = train_classifier(
            device, classifier_cnn_0, trainset, testset, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler, verbose=True, 
            pen_loss=False, pen_layer_node=None)
        
        res_list.append(res)

    for key in res_list[0].keys():
        try:
            res_0[key] = np.hstack([res[key] for res in res_list ])
        except:
            res_0[key] = res_list[0][key]
    with open ('results/res_0.pkl', 'wb') as file:
        pickle.dump(res_0, file)        

    res_1 = {}
    res_list = []
    for i in range(samples):
        print('Sample: ', i)
        
        classifier_cnn_1 = Classifier_cnn(pen_layer_node=None, hidden_layer_node=hidden_layer_base+[pen_layer_node]).to(device)
        if (i==0):
            res = train_classifier(
                device, classifier_cnn_1, trainset, testset, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler,
                pen_loss=False, save_pen=True, logging_pen=logging_pen, pen_layer_node=None)
        else:
            res = train_classifier(
                device, classifier_cnn_1, trainset, testset, batch_size=batch_size, lr=0.0001, epochs=epochs, step_scheduler=step_scheduler, verbose=False,
                pen_loss=False, save_pen=False, pen_layer_node=None)

        res_list.append(res)

    for key in res_list[0].keys():
        try:
            res_1[key] = np.hstack([res[key] for res in res_list ])
        except:
            res_1[key] = res_list[0][key]
    with open ('results/res_1.pkl', 'wb') as file:
        pickle.dump(res_1, file)