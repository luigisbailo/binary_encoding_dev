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
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("CUDA is not available. Training on CPU.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--model', required=True)

    args = parser.parse_args()


    config_file = args.config
    results_dir = args.results_dir
    architecture_model = args.model

    configs =  parse_config(config_file)

    configs_architecture = configs['architecture']
    architecture_backbone = configs['architecture']['backbone']

    architecture = configs['architecture']
    training_hyperparams = configs['training']['hyperparams']
    # training_models = configs['training']['models']
    samples = training_hyperparams['samples']

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
    if os.path.exists(path_metrics_dir):
        shutil.rmtree(path_metrics_dir)
    os.mkdir(path_metrics_dir)


    print('Training ' + str(architecture_model) + ' architecture:')
    res_dict= {}
    results_sample = []
    
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
                    training_hyperparams=training_hyperparams, 
                    model=architecture_model, 
                    verbose=verbose).fit()
            )
    for key in results_sample[0].keys():
        try:
            res_dict[key] = np.hstack([res[key] for res in results_sample])
        except:
            res_dict[key] = res_dict[0][key]
    with open (path_metrics_dir + '/res_' + architecture_model +'.pkl', 'wb') as file:
        pickle.dump(res_dict, file)

