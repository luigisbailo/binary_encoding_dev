import numpy as np
import pickle
import sys
import os 
import shutil
import yaml
import argparse
import importlib
import glob
import matplotlib.pyplot as plt

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()

    results_dir = args.results_dir
    
    models = ['bin_enc', 'lin_pen', 'nonlin_pen', 'no_pen']
    results = {}
    lr_dict = {}
    
    for model in models:
        
        results[model]={}
        file_pattern = f'{results_dir}{"/res_"}{model}*{"pkl"}'
        files = glob.glob(file_pattern)
        ll = []
        for file_path in files:
            with open(file_path, 'rb') as file:
                res = pickle.load(file)
                ll.append((res['training_hypers']['lr'] ,np.around(res['accuracy_test_mean'],4), np.around(res['accuracy_test_std'],4) ))
        ll.sort()
        
        accuracy = []
        lr = []        
        for l in ll:
            lr.append(l[0])
            accuracy.append(l[1])
        results[model]['lr']=np.array(lr)
        results[model]['accuracy']=np.array(accuracy)

        x_lr=results[model]['lr']
        y_accuracy=results[model]['accuracy']

        window_size=3
        grad = np.gradient(y_accuracy)

        moving_average_grad = np.convolve(grad, np.ones(window_size)/window_size, mode='valid')
        pad_size = (len(grad) - len(moving_average_grad)) // 2
        moving_average_grad = np.pad(moving_average_grad, (pad_size, pad_size), mode='edge')

        best_lr = x_lr[int(np.argmax(moving_average_grad))]
        lr_dict[model] = best_lr
                
        fig, axs = plt.subplots(2,1, figsize=(15, 10))
        fig.suptitle(model)
        axs[0].plot(x_lr,y_accuracy)
        axs[0].set_xscale('log')
        axs[0].axvline(x=best_lr, color='red', linestyle='--')

        axs[1].plot(x_lr,moving_average_grad)
        axs[1].set_xscale('log')
        axs[1].axvline(x=best_lr, color='red', linestyle='--')
        plt.savefig(results_dir + '/_' + model + '.png')

        
    file_name = results_dir + '/_lr.pkl'

    with open (file_name, 'wb') as file:
        pickle.dump(lr_dict, file)

                      
                      

                          
        
        
        
        