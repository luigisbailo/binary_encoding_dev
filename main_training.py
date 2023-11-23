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
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--etf-metrics', required=False)
    args = parser.parse_args()

    config_file = args.config
    results_dir = args.results_dir
    architecture_model = args.model
    sample = args.sample
    lr = args.lr
    etf_simplex_metrics = args.etf_metrics

    configs =  parse_config(config_file)
    configs_architecture = configs['architecture']
    architecture_backbone = configs['architecture']['backbone']
    architecture = configs['architecture']
    training_hypers = configs['training']['hypers']
    name_dataset = configs['dataset']['name']    
    if lr:
        training_hypers['lr'] = float(lr)
    if not etf_simplex_metrics:
        etf_simplex_metrics=False

    torch_module= importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torch_module, name_dataset)
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torch_dataset ("./dataset", train=True, download=True, transform=transform)
    testset = torch_dataset ("./dataset", train=False, download=True, transform=transform)
    trainset_mean, trainset_std = compute_mean_std(trainset)
    transform_normalized = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std)
    ])
    trainset = torch_dataset ("./dataset", train=True, download=True, transform=transform_normalized)
    testset = torch_dataset ("./dataset", train=False, download=True, transform=transform_normalized)


    path_metrics_dir = str(results_dir)

    print('Training ' + str(architecture_model) + ' architecture:')
    res_dict= {}
    results_sample = []
    
    classifier = getattr(networks, architecture_backbone)(
        model=architecture_model,
        architecture=architecture,
        )
    if torch.cuda.is_available():
        classifier = torch.nn.DataParallel(classifier)
        classifier = classifier.to(device)
    results_sample.append(
        Trainer(device=device, 
                network=classifier, 
                trainset=trainset, 
                testset=testset, 
                training_hypers=training_hypers, 
                model=architecture_model, 
                etfsimplex_metrics=etf_simplex_metrics,
                verbose=verbose).fit()
        )

    if sample:    
        file_name = '/res_' + architecture_model + '_' + sample + '.pkl'
    else:
        file_name = '/res_' + architecture_model + '.pkl'

    with open (path_metrics_dir + file_name, 'wb') as file:
        pickle.dump(res_dict, file)

