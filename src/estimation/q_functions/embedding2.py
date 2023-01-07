import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import numpy as np
import copy
from itertools import permutations
from src.environments.generate_network import lattice, random_nearest_neighbor, contrived
from scipy.special import expit
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression, Ridge
from src.utils.misc import kl, second_order_adjacency_list

class GGCN2(nn.Module):
  def __init__(self, nfeat, J, adjacency_list, binary = True, neighbor_order=1):
    super(GGCN2, self).__init__()
    self.g1 = nn.Linear(2*J, J)
    #self.g2 = nn.Linear(J, J)
    self.h1 = nn.Linear(nfeat, J)
    self.h2 = nn.Linear(J, J)
    if binary:
      self.final1 = nn.Linear(J, 2)
    else:
      self.final1 = nn.Linear(J, 1)
    self.J = J
    if neighbor_order == 1:
        self.adjacency_list = adjacency_list
    elif neighbor_order == 2:
        self.adjacency_list = second_order_adjacency_list(adjacency_list)
    self.L = len(adjacency_list)

  def final(self, E_):
    E = self.final1(E_)
    return E
    
  def h(self, b):
    b = self.h1(b)
    b = F.relu(b)
    return b

  def g(self, b):
    b = self.g1(b)
    b = F.relu(b)
    return b

  def forward(self, X_):
    E = self.embed_recursive(X_)
    yhat = self.final(E)
    return yhat

  def embed_recursive(self, X_):
    L = X_.shape[0]
    def fk(X_, idx, k):
      if k == 1:
        return self.h(X_[idx])  
      else:
        result = torch.zeros((1, self.J))
        for i in range(len(idx)):
          idx_temp = list(copy.deepcopy(idx))
          idx1 = idx_temp[i]
          idx_temp.remove(idx1)
          idx2 = idx_temp
          X_1 = X_[idx1] ###
          X_2 = fk(X_, idx2, k-1)
          h_val = self.h(X_1)
          h_val_cat_fkm1 = torch.cat((h_val, X_2[0])) ###
          g_val = self.g(h_val_cat_fkm1)
          result += g_val / len(idx) ###
        return result
    
    E = torch.tensor([])
    for l in range(L):
      neighbor_num = len(self.adjacency_list[l])
      X_l = fk(X_, self.adjacency_list[l], neighbor_num)
      E = torch.cat((E, X_l), dim=0)
    E = F.relu(E)
    temp = torch.cat((self.h(X_), E), dim=1)
    E = self.g(temp)
    return E      

def learn_ggcn2(X_list, y_list, adjacency_list, n_epoch=100, nhid=16, batch_size=2, 
                lr=0.01, tol=0.001, binary = True, neighbor_order=1):
  model = fit_ggcn2(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, 
                    batch_size=batch_size, lr=lr, tol=tol, binary=binary, neighbor_order=neighbor_order)

  def embedding_wrapper(X_):
    X_ = torch.FloatTensor(X_)
    E = model.embed_recursive(X_).detach().numpy()
    return E

  def model_wrapper(X_):
    X_ = torch.FloatTensor(X_)
    outcome = model.forward(X_)
    if binary == True:
      yhat = F.softmax(outcome, dim=1)[:, -1].detach().numpy()
    else:
      yhat = outcome[:, 0].detach().numpy()
    return yhat
    
  return embedding_wrapper, model_wrapper

def fit_ggcn2(X_list, y_list, adjacency_list, n_epoch=100, nhid=16, 
              batch_size=5, lr=0.01, tol=0.001, binary=True, neighbor_order=1):
  p = X_list[0].shape[1]
  T = len(X_list)
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.FloatTensor(y) for y in y_list]

  model = GGCN2(nfeat=p, J=nhid, adjacency_list=adjacency_list, neighbor_order=neighbor_order, binary=binary)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  if binary:
    criterion = nn.CrossEntropyLoss()
  else:
    criterion = nn.MSELoss()

  for epoch in range(n_epoch):
    avg_acc_train = 0.
    batch_ixs = np.random.choice(T, size=batch_size)

    X_batch = [X_list[ix] for ix in batch_ixs]
    y_batch = [y_list[ix] for ix in batch_ixs]

    for X, y, ix in zip(X_batch, y_batch, batch_ixs):
      model.train()
      optimizer.zero_grad()
      X = Variable(X)
      if binary:
        y = Variable(y).long()
      else:
        y = Variable(y).unsqueeze(1)
    
      output = model(X)
      loss_train = criterion(output, y)

      regular = 0.
      Lambda = 0.
      for param in model.parameters():
        regular += torch.norm(param, 1)
      loss_train += Lambda*regular

      #print(f'regular: {regular}')
      #print(f'loss_train: {loss_train}')
      loss_train.backward()
      optimizer.step()
 
  if binary:
    final_acc_train = 0.
    for X_, y_ in zip(X_list, y_list):
      yhat = F.softmax(model(X_), dim=1)[:, 1]
      acc = ((yhat > 0.5) == y_).float().mean().detach().numpy()
      final_acc_train += acc / T
    print('final_acc_train: {:.4f}'.format(final_acc_train))
  else:
    final_mse_train = 0.
    for X_, y_ in zip(X_list, y_list):
      yhat = model(X_)[:, 0]
      acc = ((yhat - y_)**2).float().mean().detach().numpy()
      final_mse_train += acc / T
    print('final_mse_train: {:.4f}'.format(final_mse_train))
  return model

if __name__ == "__main__":
  # Test
  # grid_size = 100
  # adjacency_mat = random_nearest_neighbor(grid_size)
  # #adjacency_mat = lattice(grid_size)
  # adjacency_list = [[j for j in range(grid_size) if adjacency_mat[i, j]] for i in range(grid_size)]
  # neighbor_counts = adjacency_mat.sum(axis=1)
  # n = 10
  # n_epoch = 200
  # X_list = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(n)])
  # y_probs_list = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(grid_size)]) for X in X_list])
  # y_list = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in y_probs_list])

  # # Holdout data
  # X_list_holdout = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(2)])
  # y_probs_list_holdout = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(grid_size)]) for X in
  #                                  X_list_holdout])
  # y_list_holdout = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in
  #                            y_probs_list_holdout])


  # print('Fitting ggcn')
  # _, model_ = learn_ggcn2(X_list, y_list, adjacency_list, n_epoch=50, nhid=16, batch_size=1)

  # oracle_mean = 0.
  # ggcn_mean = 0.
  # baseline_mean = 0.
  # ggcn_acc = 0.
  # ggcn_dist = 0.
  # for X, yp, y in zip(X_list_holdout, y_probs_list_holdout, y_list_holdout):
  #   y_hat_oracle = (yp > 0.5)
  #   y_hat_ggcn = (model_(X) > 0.5)
  #   y_hat_baseline = (np.mean(y) > 0.5)
  #   oracle_mean += (y_hat_oracle == y).mean() / len(y_list_holdout)
  #   ggcn_mean += (y_hat_ggcn == y).mean() / len(y_list_holdout)
  #   baseline_mean += (y_hat_baseline == y).mean() / len(y_list_holdout)
  #   ggcn_acc += abs(model_(X) - yp).sum() / len(y_list_holdout)
  #   ggcn_dist += kl(model_(X), yp)/ len(y_list_holdout)
  # print(f'oracle: {oracle_mean} ggcn: {ggcn_mean} baseline: {baseline_mean}')
  # print(f'ggcn_acc: {ggcn_acc} ggcn_dist: {ggcn_dist}')

  # clf = LogisticRegression()
  # clf.fit(np.vstack(X_list), np.hstack(y_list))
  # oracle_mean1 = 0.
  # clf_mean = 0.
  # baseline_mean1 = 0.
  # clf_acc = 0.
  # clf_dist = 0.
  # for X, yp, y in zip(X_list_holdout, y_probs_list_holdout, y_list_holdout):
  #   y_hat_oracle = (yp > 0.5)
  #   y_hat_clf = (clf.predict_proba(X)[:, 1] > 0.5)
  #   y_hat_baseline = (np.mean(y) > 0.5)
  #   oracle_mean1 += (y_hat_oracle == y).mean() / len(y_list_holdout)
  #   clf_mean += (y_hat_clf == y).mean() / len(y_list_holdout)
  #   baseline_mean1 += (y_hat_baseline == y).mean() / len(y_list_holdout)
  #   clf_acc += abs(clf.predict_proba(X)[:, 1] - yp).sum() / len(y_list_holdout)
  #   clf_dist += kl(clf.predict_proba(X)[:, 1], yp)/ len(y_list_holdout)
  # print(f'oracle: {oracle_mean1} clf: {clf_mean} baseline: {baseline_mean1}')
  # print(f'clf_acc: {clf_acc} clf_dist: {clf_dist}')


  # Test
  grid_size = 100
  adjacency_mat = random_nearest_neighbor(grid_size)
  adjacency_list = [[j for j in range(grid_size) if adjacency_mat[i, j]] for i in range(grid_size)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  n = 10
  n_epoch = 50
  X_list = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(n)])
  y_list = np.array([np.array([np.sum(X[adjacency_list[l]]) for l in range(grid_size)]) for X in X_list])

  # Holdout data
  X_list_holdout = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(2)])
  y_list_holdout = np.array([np.array([np.sum(X[adjacency_list[l]]) for l in range(grid_size)]) for X in X_list_holdout])

  print('Fitting ggcn')
  _, model_ = learn_ggcn2(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=16, binary = False)
  mse = 0.
  base = 0.
  for X, y in zip(X_list_holdout, y_list_holdout):
    y_hat_ggcn = model_(X)
    mse += ((y_hat_ggcn - y)**2).mean() / len(y_list_holdout)
    base += (y**2).mean() / len(y_list_holdout)
    #print(y_hat_ggcn)
  

  reg = Ridge()
  reg.fit(np.vstack(X_list), np.hstack(y_list))
  mse1 = 0.
  for X, y in zip(X_list_holdout, y_list_holdout):
    y_hat_linear = reg.predict(X)
    mse1 += ((y_hat_linear - y)**2).mean() / len(y_list_holdout)
  print(f'mse-ggcn: {mse} mse-l:{mse1} base:{base}')