import numpy as np
import torch
import torch.nn as nn
import scipy
from sklearn.mixture import GaussianMixture

class Trainer ():
    
    def __init__(self, device, network, trainset, testset, hyper_train, binenc_loss=True, verbose=True):

        self.device = device
        self.network = network
        self.trainset = trainset
        self.testset = testset
        self.binenc_loss = binenc_loss
        self.verbose = verbose
        self.batch_size = hyper_train['batch_size']
        self.epochs = hyper_train['epochs']
        self.step_scheduler = hyper_train['step_scheduler']
        self.lr = hyper_train['lr']
        self.loss_pen_factor = hyper_train['loss_pen_factor']
        self.logging_pen = hyper_train['logging_pen']
        self.logging = hyper_train['logging']

    def fit (self):

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True )
        
        opt = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.step_scheduler, gamma=0.5)
        res_list = []
        res_training = {}

        for epoch in range (self.epochs):
            for x,y in trainloader:
                x=x.to(self.device)
                y=y.to(self.device)
                opt.zero_grad()
                y_pred, pen_layer= self.network(x)
    
                loss_class = nn.CrossEntropyLoss(reduction='mean')(y_pred,y)
                loss = loss_class 
                if self.binenc_loss:
                    loss_pen = torch.exp(nn.functional.mse_loss(pen_layer, torch.zeros(pen_layer.shape).to(self.device), reduction='mean'))
                    loss = loss + self.loss_pen_factor*loss_pen
                    
                loss.backward()
                opt.step()
            
            if epoch%self.logging==0 :
                if self.verbose:
                    print('Epoch: ', epoch)
                save_pen = False
                if self.logging_pen>0 and epoch%self.logging_pen==0:
                    save_pen = True

                res_epoch = self.get_metrics(save_pen)

                res_list.append(res_epoch)

        for key in res_list[0].keys():
            res_training[key] = np.vstack([res_epoch [key] for res_epoch in res_list])
        
        return res_training
    


    def get_metrics(self, save_pen, get_encoding=False):

        res = {}
        self.network.eval()
        with torch.no_grad():

            loader = torch.utils.data.DataLoader (self.trainset, batch_size=10000)
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



            wclass_variation = np.trace(np.matmul(sigma_w,scipy.linalg.pinv(sigma_b)))/len(mean_class_cent)
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

            if self.network.pen_lin_nodes:  
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

            loader = torch.utils.data.DataLoader (self.testset, batch_size=10000)
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

            if self.verbose and self.network.pen_lin_nodes:
                print(np.around(accuracy_train,5), np.around(accuracy_test,5), np.around(purity, 6), '---',
                        np.around(sigma_w, 5), np.around(wclass_variation,5), np.around(equiangular, 5), np.around(maxangle, 5), np.around(equinorm, 5), '---', 
                        np.around(score,5), np.around(stds,5), np.around(peak_dist_train,5))
            elif self.verbose:
                print( np.around(accuracy_train,5), np.around(accuracy_test,5), np.around(purity, 6), '---',
                        np.around(sigma_w, 5), np.around(wclass_variation,5), np.around(equiangular, 5), np.around(maxangle, 5), np.around(equinorm, 5))

        
        return res