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
from sklearn.linear_model import Ridge
from src.utils.misc import kl, second_order_adjacency_list
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"


class GGCN(nn.Module):
  """
  Generalized graph convolutional network (not sure yet if it's a generalization strictly speaking).
  """
  def __init__(self, nfeat, J, adjacency_lst, neighbor_subset_limit=2, samples_per_k=None, neighbor_order=2):
    super(GGCN, self).__init__() 
    if neighbor_subset_limit > 1:
      self.g1 = nn.Linear(2*J, J)
      # self.g2 = nn.Linear(J, J)
    self.h1 = nn.Linear(nfeat, J)
    self.h2 = nn.Linear(J, J)
    self.final1 = nn.Linear(J, 2)
    self.neighbor_subset_limit = neighbor_subset_limit
    self.J = J
    self.samples_per_k = samples_per_k 
    if neighbor_order == 1:
      self.adjacency_list = adjacency_lst
    elif neighbor_order == 2:
      self.adjacency_list = second_order_adjacency_list(adjacency_lst)
    self.L = len(adjacency_lst)

  def final(self, E_):
    E = self.final1(E_)
    # E = F.sigmoid(E)
    return E
  
  def h(self, b):
    b = self.h1(b)
    b = F.relu(b)
    #b = self.h2(b)
    #b = F.relu(b)
    return b
  
  def g(self, bvec):
    bvec = self.g1(bvec)
    bvec = F.relu(bvec)
    # bvec = self.g2(bvec)
    # bvec = F.relu(bvec)
    return bvec

  def forward(self, X_, Subset_num=None, train=True):
    return self.forward_recursive_vec(X_, Subset_num=Subset_num, train=train)

  def forward_recursive_vec(self, X_, Subset_num=None, train=True):
    E = self.embed_recursive_vec(X_, Subset_num=Subset_num, train=train)
    yhat = self.final(E)
    return yhat

  def sample_indices_for_recursive(self):
    L = self.L
    # Collect permutations
    self.permutations_all = {k: np.zeros((L, k, self.samples_per_k)) for k in range(2, self.neighbor_subset_limit + 1)}
    self.where_k_neighbors = {k: [] for k in range(2, self.neighbor_subset_limit + 1)}
    for l in range(L):
      neighbors_l = np.append(self.adjacency_list[l], [l])
      N_l = np.min((len(neighbors_l), self.neighbor_subset_limit))
      for k in range(2, N_l + 1):
        permutations_k = list(permutations(neighbors_l, int(k))) 
        self.where_k_neighbors[k].append(l)
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False) 
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        self.permutations_all[k][l, :] = np.array(permutations_k).T

  def embed_recursive_vec(self, X_, Subset_num=None, train=True):
    L = X_.shape[0]
    self.sample_indices_for_recursive() 
    def fk(b, k):
      if k == 1:
        return self.h(b) 
      else:
        result = torch.zeros((L, self.J))
        permutations_k = self.permutations_all[k]
        where_k_neighbors_ = self.where_k_neighbors[k]
        for perm_ix in range(self.samples_per_k):
          permutations_k_perm_ix = permutations_k[:, :, perm_ix]
          # X_1 = torch.tensor(X_[permutations_k_perm_ix[where_k_neighbors_, 0]])
          X_1 = X_[permutations_k_perm_ix[where_k_neighbors_, 0]] 
          X_lst = np.column_stack([X_[permutations_k_perm_ix[where_k_neighbors_, ix]] for ix in range(1, k)])
          X_lst = torch.tensor(X_lst)
          fkm1_val = fk(X_lst, k-1)
          h_val = self.h(X_1)
          h_val_cat_fkm1_val = torch.cat((h_val, fkm1_val), dim=1)
          g_val = self.g(h_val_cat_fkm1_val)
          # g_val = h_val + fkm1_val
          result += g_val / self.samples_per_k
        return result

    E = fk(X_, self.neighbor_subset_limit)
    E = F.relu(E)
    temp = torch.cat((self.h(X_), E), dim=1)
    E = self.g(temp)
    if Subset_num is not None:
      if train:
        E = E[0:Subset_num]
      else:
        E = E[Subset_num:]
      
    return E 

def learn_ggcn(X_list, y_list, adjacency_list, n_epoch=200, nhid=16, batch_size=1,
               neighbor_subset_limit=2, samples_per_k=6, lr=0.01, tol=0.001,
               neighbor_order=2, LAMBDA=0.0005, Subset_num=None):

  model = fit_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size,
                   neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k, 
                   lr=lr, tol=tol, neighbor_order=neighbor_order, LAMBDA=LAMBDA, 
                   Subset_num=Subset_num)

  def embedding_wrapper(X_): 
    X_ = torch.FloatTensor(X_)
    E = model.embed_recursive_vec(X_).detach().numpy() 
    return E

  def model_wrapper(X_, Subset_num=0, train=False):  
    X_ = torch.FloatTensor(X_)
    outcome = model.forward_recursive_vec(X_, Subset_num=Subset_num, train=train) 
    yhat = F.softmax(outcome, dim=1)[:, -1].detach().numpy()
    return yhat

  return embedding_wrapper, model_wrapper

def fit_ggcn(X_list, y_list, adjacency_list, n_epoch=100, nhid=100, batch_size=5,
             neighbor_subset_limit=2, samples_per_k=6, lr=0.01, tol=0.001,
             neighbor_order=2, LAMBDA=0.001, Subset_num=None):
  # Specify model
  p = X_list[0].shape[1] 
  T = len(X_list) 
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.FloatTensor(y) for y in y_list]

  model = GGCN(nfeat=p, J=nhid, adjacency_lst=adjacency_list, neighbor_subset_limit=neighbor_subset_limit,
               samples_per_k=samples_per_k, neighbor_order=neighbor_order)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  
  prev_avg_acc_train = 100
  # Train
  for epoch in range(n_epoch):
    avg_acc_train = 0.
    batch_ixs = np.random.choice(T, size=batch_size)

    X_batch = [X_list[ix] for ix in batch_ixs]
    y_batch = [y_list[ix] for ix in batch_ixs]

    for X, y, ix in zip(X_batch, y_batch, batch_ixs):
      model.train()
      optimizer.zero_grad() 
      X = Variable(X)
      y = Variable(y).long() 

      output = model(X, Subset_num=Subset_num, train=True)
      loss_train = criterion(output, y[0:Subset_num])
    
      regular = 0
      #L1 regularization
      #LAMBDA = 0.0005
      for param in model.parameters():
        regular += torch.norm(param, 1)
      loss_train += LAMBDA*regular
      
      #print(f'regularization: {regular} loss: {loss_train}')
      loss_train.backward() 
      optimizer.step() 

    # Evaluate loss
    # for X_, y_ in zip(X_list, y_list):
    #   yhat = model(X_, adjacency_list)[:, 0]
    #   acc = ((yhat - y_)**2).float().mean().detach().numpy()
    #   avg_acc_train += acc / T

    # Break if change in accuracy is sufficiently small  
    # if epoch > 0:
    #   relative_acc_diff = np.abs(prev_avg_acc_train - avg_acc_train) / avg_acc_train
    #   if relative_acc_diff < tol:
    #     break

#   final_acc_train = 0.
#   for X_, y_ in zip(X_list, y_list):
#     output = model(X_, Subset_num=Subset_num, train=True)
#     yhat = F.softmax(output, dim=1)[:, 1]
#     acc = ((yhat > 0.5) == y_[0:Subset_num]).float().mean().detach().numpy()
#     final_acc_train += acc / T
  #print('final_acc_train: {:.4f}'.format(final_acc_train))


  return model

def tune_ggcn(X_list, y_list, adjacency_list, n_epoch=100, batch_size=1,
               neighbor_subset_limit=2, samples_per_k=6, num_settings_to_try=5,
               tol=0.001, neighbor_order=1):
  
  NHID_RANGE = np.linspace(5, 20, 6)
  LAMBDA_RANGE = np.logspace(-5, -2, 10)
  LR_RANGE = np.logspace(-3, -1.5, 5)
  per = 0.3
  L = X_list[0].shape[0]
  Subset_num = int((1 - per) * L)


  #Get Training and Testing Set
  
  best_settings = None
  best_model = None
  best_acc = -float('inf')
  for i in range(num_settings_to_try):
    lr = np.random.choice(LR_RANGE)
    nhid = int(np.random.choice(NHID_RANGE))
    LAMBDA = np.random.choice(LAMBDA_RANGE)
    settings = {'lr': lr, 'nhid': nhid, 'LAMBDA': LAMBDA}

    #Training
    _, model_ = learn_ggcn(X_list, y_list, adjacency_list, nhid=nhid, n_epoch=n_epoch, batch_size=batch_size,
                            neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k, tol=0.001, 
                            neighbor_order=neighbor_order, lr=lr, LAMBDA=LAMBDA, Subset_num=Subset_num)
    #Testing
    acc = 0.
    for t in range(len(y_list)):
      prediction = model_(X_list[t], Subset_num=Subset_num, train=False)
      acc_temp = ((prediction > 0.5) == y_list[t][Subset_num:]).mean()
      acc += acc_temp / len(y_list)
    #print(f'loss: {loss}')

    if acc > best_acc:
      best_model = model_
      best_settings = settings
      best_acc = acc
    #print(best_settings)

  return best_model, best_settings
    


  




if __name__ == "__main__":
  # Test
  grid_size = 100
  adjacency_mat = random_nearest_neighbor(grid_size)
  adjacency_list = [[j for j in range(grid_size) if adjacency_mat[i, j]] for i in range(grid_size)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  n = 10
  n_epoch = 100
  X_list = np.array([np.random.normal(size=(grid_size, 3)) for _ in range(n)])
  y_probs_list = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(grid_size)]) for X in X_list])
  y_list = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in y_probs_list])

  # Holdout data
  X_list_holdout = np.array([np.random.normal(size=(grid_size, 3)) for _ in range(5)])
  y_probs_list_holdout = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(grid_size)]) for X in
                                   X_list_holdout])
  y_list_holdout = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in
                             y_probs_list_holdout])
#   print('Fitting GGCN')
#   _, model_ = learn_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=16, Subset_num=70)
#   oracle_mean = 0.
#   ggcn_mean = 0.
#   baseline_mean = 0.
#   ggcn_acc = 0.
#   ggcn_dist = 0.
#   for X, yp, y in zip(X_list_holdout, y_probs_list_holdout, y_list_holdout):
#     y_hat_oracle = (yp > 0.5)
#     y_hat_ggcn = (model_(X, Subset_num=70, train=False) > 0.5)
#     y_hat_baseline = (np.mean(y) > 0.5)
#     oracle_mean += (y_hat_oracle == y).mean() / len(y_list_holdout)
#     ggcn_mean += (y_hat_ggcn == y[70:]).mean() / len(y_list_holdout)
#     baseline_mean += (y_hat_baseline == y).mean() / len(y_list_holdout)
#     ggcn_acc += abs(model_(X, Subset_num=70, train=False) - yp[70:]).sum() / len(y_list_holdout)
#     ggcn_dist += kl(model_(X, Subset_num=70, train=False), yp[70:])/ len(y_list_holdout)
#   print(f'oracle: {oracle_mean} ggcn: {ggcn_mean} baseline: {baseline_mean}')
#   print(f'ggcn_acc: {ggcn_acc} ggcn_dist: {ggcn_dist}')

  print('Tune GGCN')
  model_, setting_ = tune_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, num_settings_to_try=10)

  print('Fitting GGCN')
  _, model1_ = learn_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=16, Subset_num=100, lr=0.001, LAMBDA=0.001)

  oracle_mean = 0.
  ggcn1_mean = 0.
  ggcn2_mean = 0.
  ggcn1_acc = 0.
  ggcn2_acc = 0.
  ggcn1_dist = 0.
  ggcn2_dist = 0.
  baseline_mean = 0.
  for X, yp, y in zip(X_list_holdout, y_probs_list_holdout, y_list_holdout):
    y_hat_oracle = (yp > 0.5)
    y1_hat_ggcn = (model_(X) > 0.5)
    y2_hat_ggcn = (model1_(X) > 0.5)
    y_hat_baseline = (np.mean(y) > 0.5)
    oracle_mean += (y_hat_oracle == y).mean() / len(y_list_holdout)
    ggcn1_mean += (y1_hat_ggcn == y).mean() / len(y_list_holdout)
    ggcn2_mean += (y2_hat_ggcn == y).mean() / len(y_list_holdout)
    baseline_mean += (y_hat_baseline == y).mean() / len(y_list_holdout)
    ggcn1_acc += abs(model_(X) - yp).sum() / len(y_list_holdout)
    ggcn2_acc += abs(model1_(X) - yp).sum() / len(y_list_holdout)
    ggcn1_dist += kl(model_(X), yp)/ len(y_list_holdout)
    ggcn2_dist += kl(model1_(X), yp)/ len(y_list_holdout)
  print(f'oracle: {oracle_mean} ggcn1: {ggcn1_mean} ggcn2: {ggcn2_mean} baseline: {baseline_mean}')
  print(f'ggcn1_acc: {ggcn1_acc} ggcn1_dist: {ggcn1_dist}')
  print(f'ggcn2_acc: {ggcn2_acc} ggcn2_dist: {ggcn2_dist}')
