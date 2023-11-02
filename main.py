from nn import Classifier_cnn
from trainer import train_classifier

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import sys
import os 
import shutil
import yaml
import argparse
import zipfile
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("cuda available: ", torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    configs =  parse_config(parser.parse_args().config)

    hyper_architecture = configs['hyperparams']['architecture']
    hyper_train = configs['hyperparams']['train']
    samples = configs['samples']

    name_dataset = configs['dataset']['name']    
    torch_module= importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torch_module, name_dataset)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torch_dataset ("./dataset", train=True, download=True, transform=transform)
    testset = torch_dataset ("./dataset", train=False, download=True, transform=transform)


    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results")



    print('Training binary encoding architecture:')
    res_binenc = {}
    results = []
    for i in range(samples):
        print('Sample: ', i)
        classifier = Classifier_cnn(pen_layer_node=hyper_architecture['pen_layer_node'], hidden_layer_node=hyper_architecture['hidden_layer_base']).to(device)
        results.append(
            train_classifier(device, classifier, trainset, testset, hyper_train, binenc_loss=True, save_pen=False, verbose=True)
            )
    for key in results[0].keys():
        try:
            res_binenc[key] = np.hstack([res[key] for res in results])
        except:
            res_binenc[key] = results[0][key]
    with open ('results/res_binenc.pkl', 'wb') as file:
        pickle.dump(res_binenc, file)



    print('Training linear penultimate architecture')
    res_linpen = {}
    results = []
    for i in range(samples):
        print('Sample: ', i)        
        classifier = Classifier_cnn(pen_layer_node=hyper_architecture['pen_layer_node'], hidden_layer_node=hyper_architecture['hidden_layer_base']).to(device)
        results.append(
            train_classifier(device, classifier, trainset, testset, hyper_train, binenc_loss=False, save_pen=False, verbose=True)
            )
    for key in results[0].keys():
        try:
            res_linpen[key] = np.hstack([res[key] for res in results ])
        except:
            res_linpen[key] = results[0][key]
    with open ('results/res_linpen.pkl', 'wb') as file:
        pickle.dump(res_linpen, file)

    print('Training no penultimate architecture')
    res_nopen = {}
    results = []
    for i in range(samples):    
        print('Sample: ', i)        
        classifier = Classifier_cnn(pen_layer_node=None, hidden_layer_node=hyper_architecture['hidden_layer_base']).to(device)
        results.append(train_classifier(device, classifier, trainset, testset, hyper_train, binenc_loss=False, save_pen=False, verbose=True))
    for key in results[0].keys():
        try:
            res_nopen[key] = np.hstack([res[key] for res in results ])
        except:
            res_nopen[key] = results[0][key]
    with open ('results/res_nopen.pkl', 'wb') as file:
        pickle.dump(res_nopen, file)        


    print('Training non-linear penultimate architecture')
    res_nonlinpen = {}
    results = []
    for i in range(samples):
        print('Sample: ', i)        
        classifier = Classifier_cnn(pen_layer_node=None, hidden_layer_node=hyper_architecture['hidden_layer_base']+[hyper_architecture['pen_layer_node']]).to(device)
        results.append(train_classifier(device, classifier, trainset, testset, hyper_train, binenc_loss=False, save_pen=False, verbose=True))
    for key in results[0].keys():
        try:
            res_nonlinpen[key] = np.hstack([res[key] for res in results ])
        except:
            res_nonlinpen[key] = results[0][key]
    with open ('results/res_nonlinpen.pkl', 'wb') as file:
        pickle.dump(res_nonlinpen, file)