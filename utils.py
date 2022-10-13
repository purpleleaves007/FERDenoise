import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import numbers
from tqdm import tqdm
from torchvision.transforms import functional as F1
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import scipy.stats as stats
import math
import numpy as np
import time
import pandas as pd  
from torch.autograd import Variable
"""
Define task metrics, loss functions and model trainer here.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

def track_training_loss(model, device, train_loader):
    model.eval()
    maxps = torch.zeros(7)
    maxps1 = torch.zeros(7)

    all_losses = torch.Tensor()
    all_losses1 = torch.Tensor()
    all_label = torch.Tensor()

    for batch_idx, (path, data, target) in enumerate(train_loader):
        data, target = data.to(device),target.to(device)
        feature,prediction = model(data)
        predictions = torch.nn.functional.softmax(prediction[0], dim=1)
        predictions1 = torch.nn.functional.softmax(prediction[1], dim=1)
        target1 = torch.nn.functional.one_hot(target,7)
        idx_loss = torch.cosine_similarity(predictions, target1)
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        all_label = torch.cat((all_label, target.cpu()))
        idx_loss1 = torch.cosine_similarity(predictions1, target1)
        idx_loss1.detach_()
        all_losses1 = torch.cat((all_losses1, idx_loss1.cpu()))
    for el in range(7):
        ein = all_label == el
        maxps[el] = all_losses[ein.long()].var()
        maxps1[el] = all_losses1[ein.long()].var()     

    loss_tr = all_losses.data.numpy()
    loss_tr1 = all_losses1.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 100)
    min_perc = np.percentile(loss_tr, 0)
    #loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6


    #loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4
    loss_tr1[loss_tr1>=1] = 1-10e-4
    loss_tr1[loss_tr1 <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)
    bmm_model1 = BetaMixture1D(max_iters=10)
    bmm_model1.fit(loss_tr1)

    bmm_model.create_lookup(1)
    bmm_model1.create_lookup(1)

    maxps[maxps <= 0] = 10e-4
    maxps1[maxps1 <= 0] = 10e-4
    maxps = maxps/maxps.max()
    maxps1 = maxps1/maxps1.max()

    return bmm_model, bmm_model1, bmm_model_maxLoss, bmm_model_minLoss, maxps, maxps1

def compute_probabilities_batch(sim, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):   
    batch_losses = sim
    batch_losses.detach_()
    #batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return torch.FloatTensor(B)

def comloss(p, x_pred, feature, x_output ,iter, bmm_model, 
bmm_model1, bmm_model_maxLoss, bmm_model_minLoss, maxps, maxps1):
    device = x_pred[0].device
    loss = 0.0
    criterion = nn.CrossEntropyLoss()
    j = 0
    cw = torch.zeros(len(x_pred[0]))
    cw1 = torch.zeros(len(x_pred[0]))
    cw = cw.to(device)
    cw1 = cw1.to(device)
    if iter >=1:
        maxps = maxps.to(device)
        maxps1 = maxps1.to(device)
        for i in range(len(x_output)):
            x_o = x_output[i]
            x_o = torch.tensor([x_o]).to(device)
            x_p = torch.unsqueeze(x_pred[0][i],0)
            x_p1 = torch.unsqueeze(x_pred[1][i],0)
            lf = torch.nn.functional.softmax(x_p,dim=1)
            lf1 = torch.nn.functional.softmax(x_p1,dim=1)
            ou = torch.nn.functional.one_hot(x_o,7)
            cf = torch.cosine_similarity(ou,lf)
            cf1 = torch.cosine_similarity(ou,lf1)
            B = compute_probabilities_batch(cf, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B1 = compute_probabilities_batch(cf1, bmm_model1, bmm_model_maxLoss, bmm_model_minLoss)
            B[B <= 1e-4] = 1e-4
            B[B >= 1-1e-4] = 1-1e-4
            B1[B1 <= 1e-4] = 1e-4
            B1[B1 >= 1-1e-4] = 1-1e-4
            B = B.to(device)
            B1 = B1.to(device)
            cw[i] = B
            cw1[i] = B1
    for i in range(len(x_output)):
        x_o = x_output[i]
        x_o = torch.tensor([x_o]).to(device)
        x_p = torch.unsqueeze(x_pred[0][i],0)
        x_p1 = torch.unsqueeze(x_pred[1][i],0)
        ou = torch.nn.functional.one_hot(x_o,7)
        lf = torch.nn.functional.softmax(x_p,dim=1)
        lf1 = torch.nn.functional.softmax(x_p1,dim=1)
        lflog = F.log_softmax(x_p,dim=1)
        lflog1 = F.log_softmax(x_p1,dim=1)
        if iter>=1:
            wce = 1
            c = cw[i]#/maxps[x_o]
            c1 = cw1[i]#/maxps1[x_o]
            simloss = F.mse_loss(lf,lf1)
            floss = c1*F.nll_loss(lflog, x_o)
            floss1 = c*F.nll_loss(lflog1, x_o)
            loss = loss + wce*(floss + floss1) + 5*simloss
            j=j+1
        else:
            floss = F.nll_loss(lflog, x_o) 
            floss1 = F.nll_loss(lflog1, x_o) 
            loss = loss + floss + floss1
            j=j+1
    if j==0:
        lossfi = loss +criterion(x_p, x_o)*0
    else:
        lossfi = loss/j
    return lossfi

def cross_denoising_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, btr, bte, total_epoch=200):
    train_batch = btr
    test_batch = bte
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    ones = torch.sparse.torch.eye(6)
    ones = ones.to(device)
    bestac = 50.0
    bmm_model=bmm_model_maxloss=bmm_model_minloss = maxps = maxps1 = 0
    bmm_model1 = 0
    with open("log/acc.txt", "w") as f:
        for index in range(total_epoch):
            print('\nEpoch: %d' % (index + 1))
            cost = np.zeros(24, dtype=np.float32)
            correctg = 0.0
            correcti = 0.0
            correcti1 = 0.0
            totalg = 0.0
            totali = 0.0
            multi_task_model.train()
            train_dataset = iter(train_loader)
            i=1
            criterion = nn.CrossEntropyLoss()
            for k in tqdm(range(train_batch)):
                path, train_data, train_label = train_dataset.next()
                train_data, train_label = train_data.to(device), train_label.to(device)
                feature,train_pred = multi_task_model(train_data)
                losses = comloss(path, train_pred, feature, train_label,index, bmm_model, bmm_model1, bmm_model_maxloss, bmm_model_minloss, maxps, maxps1)
                loss = losses 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predictedg = torch.max(train_pred[0].data, 1)
                totalg += train_label.size(0)
                correctg += predictedg.eq(train_label.data).cpu().sum()
                _, predictedi1 = torch.max(train_pred[1].data, 1)
                correcti1 += predictedi1.eq(train_label.data).cpu().sum()
                cost[0] = loss.item()
                cost[1] = loss.item()
                avg_cost[index, :2] += cost[:2] / train_batch
                i=i+1
            print('Epoch: {:04d} | TRAINLOSS: {:.4f} {:.4f} | TRAINACC: {:.4f} {:.4f}|'.format(index, avg_cost[index, 0], avg_cost[index, 1], 100. * correctg / totalg, 100. * correcti1 / totalg))
            scheduler.step()
            # evaluating test data
            multi_task_model.eval()
            with torch.no_grad():  # operations inside don't track history
                test_dataset = iter(test_loader)
                correctg = 0.0
                correcti = 0.0
                correcti1 = 0.0
                totalg = 0.0
                totali = 0.0
                labelte = []
                labelte1 = []
                pre = []
                pre1 = []
                for k in range(test_batch):
                    test_data, test_label = test_dataset.next()
                    test_data, test_label = test_data.to(device), test_label.to(device)
                    feature,test_pred = multi_task_model(test_data)
                    test_loss = [criterion(test_pred[0], test_label),
                            criterion(test_pred[1], test_label)]

                    cost[2] = test_loss[0].item()
                    cost[3] = test_loss[1].item()

                    avg_cost[index, 2:] += cost[2:] / test_batch

                    _, predictedg = torch.max(test_pred[0].data, 1)
                    totalg += test_label.size(0)
                    correctg += predictedg.eq(test_label.data).cpu().sum()
                    _, predictedi1 = torch.max(test_pred[1].data, 1)
                    correcti1 += predictedi1.eq(test_label.data).cpu().sum()
                    totali += test_label.size(0)
                    correcti += predictedg.eq(test_label.data).cpu().sum()
                    labelte = labelte+test_label.cpu().numpy().tolist()
                    pre = pre+predictedg.cpu().numpy().tolist()
                    labelte1 = labelte1+test_label.cpu().numpy().tolist()
                    pre1 = pre1+predictedi1.cpu().numpy().tolist()
            print('Epoch: {:04d} | TESTLOSS: {:.4f} {:.4f}| TESTACC: {:.4f} {:.4f}'
                .format(index+1,  avg_cost[index, 2], avg_cost[index, 3], 100. * correctg / totalg, 100. * correcti1 / totali))
            f.write("EPOCH=%d,accges= %.3f%%,accges1= %.3f%%" % (index + 1, 100. * correctg / totalg, 100. * correcti1 / totali))
            f.write('\n')
            f.flush()
            if (100. * correcti / totali)>bestac:
                bestac = 100. * correcti / totali
                cm = confusion_matrix(labelte, pre)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm1 = confusion_matrix(labelte1, pre1)
                cm_normalized1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
                print('Saving Model')
                torch.save(multi_task_model.state_dict(),r'model/bestmodel.pth')
                f3 = open("log/bestacc.txt", "w")
                f3.write(str(bestac))
                f3.write(str(cm_normalized))
                f3.write(str(cm_normalized1))
                f3.close()
            bmm_model, bmm_model1, bmm_model_maxloss, bmm_model_minloss, maxps, maxps1 = track_training_loss(multi_task_model,device,train_loader)