import networks
from trainer import Trainer

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

def convert_bool(dict):

    for key in dict.keys():

        if dict[key] == 'True' or dict[key] == 'true':
            dict[key] = True
        elif dict[key] == 'False' or dict[key] == 'false':
            dict[key] = False

    return dict

def compute_mean_std(dataset):

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    data = next(iter(loader))[0].numpy()

    mean = np.mean(data, axis=(0, 2, 3))
    std = np.std(data, axis=(0, 2, 3))

    return mean, std

if __name__ == '__main__':

    verbose = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("CUDA is not available. Training on CPU.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--lr-file', required=False)
    parser.add_argument('--etf-metrics', required=False)
    args = parser.parse_args()

    config_file = args.config
    results_dir = args.results_dir
    dataset_dir = args.dataset_dir
    architecture_model = args.model
    sample = args.sample
    lr = args.lr
    lr_file = args.lr_file
    etf_simplex_metrics = args.etf_metrics

    configs =  parse_config(config_file)
    architecture = configs['architecture']
    training_hypers = configs['training']['hypers']
    name_dataset = configs['dataset']['name']    
    if lr:
        if lr.startswith('.'):
            lr = '0'+lr
        training_hypers['lr'] = float(lr)
    if lr_file:
        with open(lr_file, 'rb') as file:
            lr_dict = pickle.load(file)
            training_hypers['lr']=lr_dict[architecture_model]
    if not etf_simplex_metrics:
        etf_simplex_metrics=False
    elif etf_simplex_metrics == 'True' or etf_simplex_metrics == 'true':
        etf_simplex_metrics=True
    elif etf_simplex_metrics == 'False' or etf_simplex_metrics == 'false':
        etf_simplex_metrics=False
  
    print(architecture_model)
    print('Learning rate: ', training_hypers['lr']) 
    
    torch_module= importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torch_module, name_dataset)
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torch_dataset (str(dataset_dir), train=True, download=True, transform=transform)
    trainset_mean, trainset_std = compute_mean_std(trainset)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(trainset[0][0][0][0].shape[0], padding=4),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
    ])

    trainset = torch_dataset (str(dataset_dir), train=True, download=True, transform=transform_train)
    testset = torch_dataset (str(dataset_dir), train=False, download=True, transform=transform_test)

    in_channels = trainset[0][0].shape[0]  
    num_classes = len(set(trainset.classes))
    
    path_metrics_dir = str(results_dir)

    print('Training ' + str(architecture_model) + ' architecture:')
    
    training_hypers = convert_bool(training_hypers)
    architecture['hypers'] = convert_bool(architecture['hypers'])

    classifier = getattr(networks, architecture['backbone'])(
        model=architecture_model,
        architecture=architecture,
        in_channels=in_channels,
        num_classes=num_classes
        )
    if torch.cuda.is_available():
        classifier = torch.nn.DataParallel(classifier)
        classifier = classifier.to(device)
    

    res = Trainer(device=device, 
            network=classifier, 
            trainset=trainset, 
            testset=testset, 
            training_hypers=training_hypers, 
            model=architecture_model, 
            etfsimplex_metrics=etf_simplex_metrics,
            verbose=verbose).fit()
    res['training_hypers'] = training_hypers
    res['architecture'] = architecture    

    if sample:    
        file_name = '/res_' + architecture_model + '_' + sample + '.pkl'
    else:
        file_name = '/res_' + architecture_model + '.pkl'

    with open (path_metrics_dir + file_name, 'wb') as file:
        pickle.dump(res, file)

