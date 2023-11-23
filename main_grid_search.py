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
from itertools import product

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
    parser.add_argument('--samples', required=True)

    args = parser.parse_args()

    config_file = args.config
    results_dir = args.results_dir
    architecture_model = args.model
    samples = int(args.samples)

    configs =  parse_config(config_file)

    architecture = configs['architecture']
    architecture_backbone = configs['architecture']['backbone']
    training_hypers = configs['training']['hypers']
    grid_search_hypers = configs['grid_search']['hypers']
    grid_search_patience = configs['grid_search']['patience']
    name_dataset = configs['dataset']['name']    

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
    output_file = path_metrics_dir + '/' + str(architecture_model) + '_gridsearch.txt' 


    print('Training ' + str(architecture_model) + ' architecture:')

    with open(output_file, 'a') as file:
        print(str(architecture_model) + ' architecture', file=file)

    training_hypers_grid = training_hypers
    grid_keys = list(grid_search_hypers.keys())
    grid_values = list(grid_search_hypers.values())
    combinations = product(*grid_values)

    with open(output_file, 'a') as file:
        print(*grid_keys, sep='\t', file=file)

    for combo in combinations:

        res_dict= {}
        results_sample = []
        
        grid_combination = dict(zip(grid_keys, combo))
        print(grid_combination)
        
        for key in grid_keys:
            training_hypers_grid[key] = grid_combination[key]

        for i in range(samples):
            print('Sample: ', i)
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
                        training_hypers=training_hypers_grid, 
                        model=architecture_model, 
                        verbose=verbose).fit(patience=grid_search_patience)
                )
        
        accuracy_test_convergence = [res_dict['accuracy_test'][-grid_search_patience-1] for res_dict in results_sample]
        output_line = list(grid_combination.values()) + [np.mean(accuracy_test_convergence)] + [np.std(accuracy_test_convergence)]

        with open(output_file, 'a') as file:
            print(*output_line, sep='\t', file=file)