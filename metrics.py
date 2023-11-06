import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import scipy

def get_metrics(device, network, trainset, testset, save_pen, verbose):

    res = {}
    network.eval()
    with torch.no_grad():

        loader = torch.utils.data.DataLoader (trainset, batch_size=10000)
        y_pred_set = []
        y_set = []
        pen_layer_set = []
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)
            y_pred, pen_layer= network(x)
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

        if network.pen_lin_nodes:  
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


        loader = torch.utils.data.DataLoader (testset, batch_size=10000)
        y_pred_set = []
        y_set = []
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)
            y_pred, _= network(x)
            y_set.append(y)
            y_pred_set.append(y_pred)
        y_pred_set = torch.cat(y_pred_set)
        y_set = torch.cat(y_set)

        accuracy_test = (torch.argmax(y_pred_set, dim=1)==y_set).float().mean().cpu().numpy()
        res['accuracy_test'] = accuracy_test

        if verbose and network.pen_lin_nodes:
            print(np.around(accuracy_train,5), np.around(accuracy_test,5), np.around(purity, 6), '---',
                    np.around(sigma_w, 5), np.around(wclass_variation,5), np.around(equiangular, 5), np.around(maxangle, 5), np.around(equinorm, 5), '---', 
                    np.around(score,5), np.around(stds,5), np.around(peak_dist_train,5))
        elif verbose:
            print( np.around(accuracy_train,5), np.around(accuracy_test,5), np.around(purity, 6), '---',
                    np.around(sigma_w, 5), np.around(wclass_variation,5), np.around(equiangular, 5), np.around(maxangle, 5), np.around(equinorm, 5))


    if network.pen_lin_nodes:  

        with torch.no_grad():

            loader = torch.utils.data.DataLoader (trainset, batch_size=10000)
            y_pred_set = []
            y_set = []
            pen_layer_set = []
            for x,y in loader:
                x=x.to(device)
                y=y.to(device)
                y_pred, pen_layer= network(x)
                y_set.append(y)
                y_pred_set.append(y_pred)
                pen_layer_set.append(pen_layer)
            y_pred_set = torch.cat(y_pred_set)
            y_set = torch.cat(y_set)
            pen_layer_set = torch.cat(pen_layer_set)

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

    
    return res