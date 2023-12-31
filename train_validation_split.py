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
    parser.add_argument('--losspen-funct', required=False)
    parser.add_argument('--sample', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--etf-metrics', required=False)
    args = parser.parse_args()

    config_file = args.config
    results_dir = args.results_dir
    dataset_dir = args.dataset_dir
    architecture_model = args.model
    losspen_funct = args.losspen_funct
    sample = args.sample
    lr = args.lr
    etf_simplex_metrics = args.etf_metrics

    configs =  parse_config(config_file)
    architecture = configs['architecture']
    training_hypers = configs['training']['hypers']
    name_dataset = configs['dataset']['name']    
    if lr:
        if lr.startswith('.'):
            lr = '0'+lr
        training_hypers['lr'] = float(lr)
    if losspen_funct:
        training_hypers['loss_pen_funct']=losspen_funct
        print(training_hypers['loss_pen_funct'])
    if not etf_simplex_metrics:
        etf_simplex_metrics=False
    elif etf_simplex_metrics == 'True' or etf_simplex_metrics == 'true':
        etf_simplex_metrics=True
    elif etf_simplex_metrics == 'False' or etf_simplex_metrics == 'false':
        etf_simplex_metrics=False

        
    training_hypers = convert_bool(training_hypers)
    architecture['hypers'] = convert_bool(architecture['hypers'])

    
    torch_module= importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torch_module, name_dataset)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torch_dataset (str(dataset_dir), train=True, download=True, transform=transform)
    dataset_mean, dataset_std = compute_mean_std(dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(dataset[0][0][0][0].shape[0], padding=4),
    ])
    dataset.transform = transform

    in_channels = dataset[0][0].shape[0]  
    num_classes = len(set(dataset.classes))
    
    path_metrics_dir = str(results_dir)

    print('Training ' + str(architecture_model) + ' architecture:')
    
    epochs = 20
    training_hypers['epochs']=int(epochs)
    training_hypers['logging']=int(epochs)
    
    accuracy_train = []
    accuracy_test = []
    
    for seed in range (3):        


        trainset, testset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(seed))
        
        classifier = getattr(networks, architecture['backbone'])(
            model=architecture_model,
            architecture=architecture,
            in_channels=in_channels,
            num_classes=num_classes
            )
        if torch.cuda.is_available():
            classifier = torch.nn.DataParallel(classifier)
            classifier = classifier.to(device)


        _ = Trainer(device=device, 
                network=classifier, 
                trainset=trainset, 
                testset=testset, 
                training_hypers=training_hypers, 
                model=architecture_model, 
                etfsimplex_metrics=etf_simplex_metrics,
                verbose=verbose).fit()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        dataset.transform = transform

        with torch.no_grad():
            
            classifier.eval()

            loader_train = torch.utils.data.DataLoader (trainset, batch_size=1000)
            loader_test = torch.utils.data.DataLoader (testset, batch_size=1000)


            y_pred_set = []
            y_set = []
            for x,y in loader_train:
                x=x.to(device)
                y=y.to(device)
                y_pred, _= classifier(x)
                y_set.append(y)
                y_pred_set.append(y_pred)
            y_pred_set = torch.cat(y_pred_set)
            y_set = torch.cat(y_set)
            accuracy_train.append((torch.argmax(y_pred_set, dim=1)==y_set).float().mean().cpu().numpy())        

            y_pred_set = []
            y_set = []
            for x,y in loader_test:
                x=x.to(device)
                y=y.to(device)
                y_pred, _= classifier(x)
                y_set.append(y)
                y_pred_set.append(y_pred)
            y_pred_set = torch.cat(y_pred_set)
            y_set = torch.cat(y_set)
            accuracy_test.append((torch.argmax(y_pred_set, dim=1)==y_set).float().mean().cpu().numpy())        


    res = {'accuracy_train_mean': np.mean(accuracy_train),
           'accuracy_train_std': np.std(accuracy_train),
           'accuracy_test_mean': np.mean(accuracy_test),
           'accuracy_test_std': np.std(accuracy_test)}
    res['training_hypers'] = training_hypers
    res['architecture'] = architecture    

    
    if architecture_model == 'bin_enc':
        architecture_model = architecture_model + '_' + training_hypers['loss_pen_funct']
    if sample:    
        file_name = '/res_' + architecture_model + '_' + sample + '.pkl'
    else:
        file_name = '/res_' + architecture_model + '.pkl'

    with open (path_metrics_dir + file_name, 'wb') as file:
        pickle.dump(res, file)

