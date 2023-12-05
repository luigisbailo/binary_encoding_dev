import numpy as np
import torch
import torch.nn as nn
import scipy
from sklearn.mixture import GaussianMixture
import importlib
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer ():
    
    def __init__(self, device, network, trainset, testset, training_hypers, model, etfsimplex_metrics=False, verbose=True):

        self.device = device
        self.network = network
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.etfsimplex_metrics=etfsimplex_metrics
        self.verbose = verbose
        
        self.training_hypers = training_hypers
        if model == 'bin_enc':
            self.binenc_loss = True
        else:
            self.binenc_loss = False
    
    def make_optimizer(self, lr):
        torch_module= importlib.import_module("torch.optim")

        if (self.training_hypers['optimizer'] == 'SGD'):
            self.opt = getattr(torch_module, self.training_hypers['optimizer'])(self.network.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        elif (self.training_hypers['optimizer'] == 'Adam'):
            if self.training_hypers['amsgrad']:
                self.opt = getattr(torch_module, self.training_hypers['optimizer'])(self.network.parameters(), lr=lr, amsgrad=True)
            else: 
                self.opt = getattr(torch_module, self.training_hypers['optimizer'])(self.network.parameters(), lr=lr, amsgrad=False)
        elif (self.training_hypers['optimizer'] == 'AdamW'):
            if self.training_hypers['amsgrad']:
                self.opt = getattr(torch_module, self.training_hypers['optimizer'])(self.network.parameters(), lr=lr, amsgrad=True)
            else:
                self.opt = getattr(torch_module, self.training_hypers['optimizer'])(self.network.parameters(), lr=lr, amsgrad=False)                
        else:
            print('Error: Optimizer not recognized')
            sys.exit(1)

        if self.training_hypers['step_scheduler']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.training_hypers['step_scheduler'], gamma=self.training_hypers['gamma'])
            


    def find_lr (self, lrs, epochs=2, plot=False, verbose=False):
        
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.training_hypers['batch_size'], shuffle=True)
        loss_values = []
        
        if verbose:
            print(str("lrs"), str("loss"), sep='\t')
            
        for lr in lrs:
            self.make_optimizer(lr=lr)
            self.network.module.reset_parameters()
            loss_pen_factor = self.training_hypers['loss_pen_factor']

            for epoch in range (epochs):
                for x,y in trainloader:
                    x=x.to(self.device)
                    y=y.to(self.device)

                    self.opt.zero_grad()
                    y_pred, pen_layer= self.network(x)

                    loss_class = nn.CrossEntropyLoss(reduction='mean')(y_pred,y)
                    loss = loss_class 
                    if self.binenc_loss:
                        if self.training_hypers['loss_pen_funct'] == 'exp_mse':
                            loss_pen = torch.exp(nn.functional.mse_loss(pen_layer, torch.zeros(pen_layer.shape).to(self.device), reduction='mean'))
                        elif self.training_hypers['loss_pen_funct'] == 'mse':
                            loss_pen = nn.functional.mse_loss(pen_layer, torch.zeros(pen_layer.shape).to(self.device), reduction='mean')
                        else:
                            print("Error: penultimate loss not available")
                            sys.exit(1)
                        loss = loss + loss_pen*loss_pen_factor
                    loss.backward()
                    self.opt.step()

                loss_pen_factor = loss_pen_factor * self.training_hypers['loss_pen_factor_gamma']
            
            loss_class = loss_class.detach().cpu().numpy()
            if verbose:
                print(lr, loss_class, sep='\t')
            loss_values.append(loss_class)
        
        if plot:
            plt.xscale('log')  
            plt.plot(lrs, loss_values )
        
        return loss_values
        
    
    def fit (self, patience=None):

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.training_hypers['batch_size'], shuffle=True)
        
        res_list = []
        res_training = {}

        if patience==None:
            patience=self.training_hypers['epochs']
        best_accuracy_test = 0
        counter = 0
        
        self.make_optimizer(lr=self.training_hypers['lr'])
        loss_pen_factor = self.training_hypers['loss_pen_factor']
        
        for epoch in range (self.training_hypers['epochs']):
            for x,y in trainloader:
                x=x.to(self.device)
                y=y.to(self.device)
                
                self.opt.zero_grad()
                y_pred, pen_layer= self.network(x)
    
                loss_class = nn.CrossEntropyLoss(reduction='mean')(y_pred,y)
                loss = loss_class 
                if self.binenc_loss:
                    if self.training_hypers['loss_pen_funct'] == 'exp_mse':
                        loss_pen = torch.exp(nn.functional.mse_loss(pen_layer, torch.zeros(pen_layer.shape).to(self.device), reduction='mean'))
                    elif self.training_hypers['loss_pen_funct'] == 'mse':
                        loss_pen = nn.functional.mse_loss(pen_layer, torch.zeros(pen_layer.shape).to(self.device), reduction='mean')
                    else:
                        print("Error: penultimate loss not available")
                        sys.exit(1)
                    loss = loss + loss_pen*loss_pen_factor
                loss.backward()
                self.opt.step()

            loss_pen_factor = loss_pen_factor * self.training_hypers['loss_pen_factor_gamma']
            
            if epoch%self.training_hypers['logging']==0 and epoch!=0:
                if self.verbose:
                    print('Epoch: ', epoch)
                save_pen = False
                if self.training_hypers['logging_pen']>0 and epoch%self.training_hypers['logging_pen']==0:
                    save_pen = True

                res_epoch = self.get_metrics(save_pen, etfsimplex_metrics=self.etfsimplex_metrics)

                accuracy_test = res_epoch['accuracy_test']
                if accuracy_test > best_accuracy_test:
                    best_accuracy_test = accuracy_test
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    break

                res_list.append(res_epoch)

        for key in res_list[0].keys():
            res_training[key] = np.vstack([res_epoch [key] for res_epoch in res_list])
        
        return res_training
    


    def get_metrics(self, save_pen, get_encoding=False, etfsimplex_metrics=True):

        res = {}
        self.network.eval()
        with torch.no_grad():

            loader = torch.utils.data.DataLoader (self.trainset, batch_size=2000)
            y_pred_set = []
            y_set = []
            pen_layer_set = []
            for x,y in loader:
                x=x.to(self.device)
                y=y.to(self.device)
                y_pred, pen_layer= self.network(x)
                y_set.append(y)
                y_pred_set.append(y_pred)
                pen_layer_set.append(pen_layer)
            y_pred_set = torch.cat(y_pred_set)
            y_set = torch.cat(y_set)
            pen_layer_set = torch.cat(pen_layer_set)
            loss_train = nn.CrossEntropyLoss(reduction='mean')(y_pred_set, y_set).detach().cpu().numpy()
            accuracy_train = (torch.argmax(y_pred_set, dim=1)==y_set).float().mean().cpu().numpy()        
            res['loss_train']=loss_train
            res['accuracy_train'] = accuracy_train
            if save_pen:
                res['pen']=pen_layer_set.cpu().numpy()
    
            if etfsimplex_metrics:
                mean_class_cent = []
                sigma_w = []
                sigma_b_class = []
                global_mean = np.mean(pen_layer_set.cpu().numpy(), axis=0)

                for class_label in np.unique(y.cpu()):
                    sel = np.argmax(y_pred_set.cpu().numpy(),axis=1)==class_label
                    if (np.sum(sel)>0):
                        mean_class = np.mean(pen_layer_set.cpu().numpy()[sel], axis=0)
                        mean_class_cent.append(mean_class-global_mean)
                        sigma_w_class = np.cov((pen_layer_set.cpu().numpy()[sel]-mean_class).T)

                        sigma_w.append(sigma_w_class)
                        sigma_b_class.append((mean_class-global_mean))
                sigma_w = np.mean(sigma_w, axis=0)
                sigma_b = np.cov(np.array(sigma_b_class).T)

                cosine = []
                cosine_max = []
                for i in range(0, len(mean_class_cent)):
                    for j in range (i, len(mean_class_cent)):
                        cosine_max.append ( np.abs(np.dot(mean_class_cent[i], mean_class_cent[j])/ np.linalg.norm(mean_class_cent[i]) /  np.linalg.norm(mean_class_cent[j]) + 1./9 ))
                        if i!=j:                        
                            cosine.append( np.dot(mean_class_cent[i], mean_class_cent[j])/ np.linalg.norm(mean_class_cent[i]) /  np.linalg.norm(mean_class_cent[j]))


                try:
                    wclass_variation = np.trace(np.matmul(sigma_w,scipy.linalg.pinv(sigma_b)))/len(mean_class_cent)
                except:
                    wclass_variation = 0
                equiangular = np.std(cosine) 
                maxangle = np.mean(cosine_max)
                equinorm = np.std( np.linalg.norm(mean_class_cent, axis=1))/np.mean( np.linalg.norm(mean_class_cent, axis=1))
                sigma_w = np.mean(sigma_w)
                res['wclass_variation'] = wclass_variation
                res['equiangular'] = equiangular
                res['maxangle'] = maxangle
                res['equinorm'] = equinorm
                res['sigma_w'] = sigma_w


                score = []
                stds = []
                purity = []
                peak_dist_train = []

                if self.model == 'bin_enc' or self.model == 'lin_pen':  
                    for d in range (pen_layer.shape[1]):
                        gmm = GaussianMixture(n_components=2)
                        gmm.fit(pen_layer_set[:,d].cpu().reshape(-1,1))
                        score.append(gmm.score(pen_layer_set[:,d].cpu().reshape(-1,1)))
                        means = gmm.means_.flatten()
                        std = np.sqrt(gmm.covariances_).flatten()
                        stds.append(std)
                        peak_dist_train.append(np.abs(means[0]-means[1])/np.mean(std))

                    score = np.mean(score)
                    stds = np.mean(stds)
                    peak_dist_train = np.mean(peak_dist_train)
                    res['score'] = score
                    res['stds'] = stds  
                    res['peak_dist_train'] = peak_dist_train

                for d in range (pen_layer.shape[1]):
                    for class_label in np.unique(y.cpu()):
                        sel = np.argmax(y_pred_set.cpu().numpy(),axis=1)==class_label
                        if (np.sum(sel) > 0):
                            fract = (pen_layer_set[sel][:,d].cpu()>0).cpu().numpy().mean()
                            purity.append(max(fract, 1-fract))

                purity = np.mean(purity) 
                res ['purity'] = purity
            
            if get_encoding:  

                encoding = -1*np.ones([len(np.unique(y.cpu())), pen_layer_set.shape[1]])       
                for d in range (pen_layer_set.shape[1]):
                    gmm = GaussianMixture(n_components=2)
                    gmm.fit(pen_layer_set[:,d].cpu().reshape(-1,1))
                    means = gmm.means_.flatten()
                    for class_label in np.unique(y.cpu()):
                        sel = np.argmax(y_pred_set.cpu().numpy(),axis=1)==class_label
                        if (np.sum(sel) > 0):
                            fract = (pen_layer_set[sel][:,d].cpu()>0).cpu().numpy().mean()
                            encoding [class_label,d] = np.where(fract<0.5,0,1)

                res['encoding'] = encoding

            loader = torch.utils.data.DataLoader (self.testset, batch_size=2000)
            y_pred_set = []
            y_set = []
            for x,y in loader:
                x=x.to(self.device)
                y=y.to(self.device)
                y_pred, _= self.network(x)
                y_set.append(y)
                y_pred_set.append(y_pred)
            y_pred_set = torch.cat(y_pred_set)
            y_set = torch.cat(y_set)

            accuracy_test = (torch.argmax(y_pred_set, dim=1)==y_set).float().mean().cpu().numpy()
            res['accuracy_test'] = accuracy_test
            
            if self.model == 'bin_enc' or self.model == 'lin_pen':
                if etfsimplex_metrics:
                    print(np.around(accuracy_train,5), np.around(accuracy_test,5), np.around(purity, 6), '---',
                            np.around(sigma_w, 5), np.around(wclass_variation,5), np.around(equiangular, 5), np.around(maxangle, 5), np.around(equinorm, 5), '---', 
                            np.around(score,5), np.around(stds,5), np.around(peak_dist_train,5))
                else:
                    print(np.around(accuracy_train,5), np.around(accuracy_test,5))


            elif self.verbose:
                if etfsimplex_metrics:
                    print( np.around(accuracy_train,5), np.around(accuracy_test,5), np.around(purity, 6), '---',
                            np.around(sigma_w, 5), np.around(wclass_variation,5), np.around(equiangular, 5), np.around(maxangle, 5), np.around(equinorm, 5))
                else:
                    print( np.around(accuracy_train,5), np.around(accuracy_test,5) )


        
        return res