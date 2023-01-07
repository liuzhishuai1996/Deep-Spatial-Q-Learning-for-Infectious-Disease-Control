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


class GGCN1(nn.Module):
  """
  Generalized graph convolutional network (not sure yet if it's a generalization strictly speaking).
  """
  def __init__(self, nfeat, J, adjacency_lst, neighbor_subset_limit=2, samples_per_k=None, recursive=False, dropout=0.0, neighbor_order=2):
    super(GGCN1, self).__init__() #继承父类init
    if neighbor_subset_limit > 1 and recursive:
      self.g1 = nn.Linear(2*J, J)
      # self.g2 = nn.Linear(J, J)
    self.h1 = nn.Linear(nfeat, J)
    self.h2 = nn.Linear(J, J)
    self.final1 = nn.Linear(J, 1) #embedding feature + raw feature 
    self.dropout_final = nn.Dropout(p=dropout)
    self.neighbor_subset_limit = neighbor_subset_limit
    self.J = J
    self.samples_per_k = samples_per_k 
    self.recursive = recursive 
    if neighbor_order == 1:
      self.adjacency_list = adjacency_lst
    elif neighbor_order == 2:
      self.adjacency_list = second_order_adjacency_list(adjacency_lst)
    self.L = len(adjacency_lst)

  def final(self, E_, train=True):
    E = self.final1(E_)
    # E = F.sigmoid(E)
    return E
  
  #h用了两层
  def h(self, b):
    b = self.h1(b)
    b = F.relu(b)
    #b = self.h2(b)
    #b = F.relu(b)
    return b
  
  #g用了一层
  def g(self, bvec):
    bvec = self.g1(bvec)
    bvec = F.relu(bvec)
    # bvec = self.g2(bvec)
    # bvec = F.relu(bvec)
    return bvec

  def forward(self, X_, adjacency_lst, location_subset=None):
    if self.recursive:
      # return self.forward_recursive(X_, adjacency_lst)
      return self.forward_recursive_vec(X_, location_subset=location_subset)
    else:
      return self.forward_simple(X_, adjacency_lst) 

  def forward_simple(self, X_, adjacency_lst):
    # Average a function of permutations of all neighbors, rather than all subsets of all neighbors

    L = self.L
    final_ = torch.tensor([])
    X_ = torch.tensor(X_).float()
    for l in range(L):
      neighbors_l = adjacency_lst[l] + [l] 
      N_l = len(neighbors_l)

      def fk(k):
        permutations_k = list(permutations(neighbors_l, int(k)))
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        # result = torch.zeros(self.J)
        result = torch.zeros(1)
        for perm in permutations_k:
          # ToDo: just takes the first element of the permutation, makes no sense
          x_l1 = torch.tensor(X_[perm[0], :])
          h_val = self.h(x_l1) #还没编完 
          result += h_val / len(permutations_k)
        return result

      E_l = fk(N_l)
      # final_l = E_l
      final_l = self.final(E_l) 
      final_ = torch.cat((final_, final_l))

    params = list(self.parameters())
    yhat = F.sigmoid(final_)
    return yhat

  def forward_recursive_vec(self, X_, location_subset=None, train=True):
    E = self.embed_recursive_vec(X_, locations_subset=location_subset)
    yhat = self.final(E, train=train)
    return yhat

  def sample_indices_for_recursive(self, locations_subset=None):
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

  #返回J+p维特征
  def embed_recursive_vec(self, X_, locations_subset=None):
    L = X_.shape[0]
    self.sample_indices_for_recursive() 
    def fk(b, k):
      if k == 1:
        return self.h(b) #对应论文中的f^1
      else:
        result = torch.zeros((L, self.J))
        permutations_k = self.permutations_all[k]
        where_k_neighbors_ = self.where_k_neighbors[k]
        for perm_ix in range(self.samples_per_k):
          permutations_k_perm_ix = permutations_k[:, :, perm_ix]
          # ToDo: indices in where_k_neighbors_ will be wrong for k < neighbor_subset_limit, because X_ shrinks 没太看懂 先不管了 neighbor_subset_limit=2暂时不会出错
          # X_1 = torch.tensor(X_[permutations_k_perm_ix[where_k_neighbors_, 0]])
          X_1 = X_[permutations_k_perm_ix[where_k_neighbors_, 0]] # g的第一个位置上的值 但这里的代码我看不懂
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
    if locations_subset is not None:
      E = E[locations_subset]
    return E #返回特征 特征是J+nfeat维的

  def forward_recursive(self, X_, adjacency_lst):
    L = X_.shape[0]
    final_ = torch.tensor([])
    X_ = torch.tensor(X_)
    # ToDo: vectorize
    for l in range(L):
      neighbors_l = adjacency_lst[l] + [l]
      N_l = np.min((len(neighbors_l), self.neighbor_subset_limit))

      def fk(b, k):
        permutations_k = list(permutations(neighbors_l, int(k)))
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        if k == 1:
          return self.h(b[0])
        else:
          result = torch.zeros(self.J)
          for perm in permutations_k:
            x_l1 = torch.tensor(X_[perm[0], :])
            x_list = X_[perm[1:], :]
            fkm1_val = fk(x_list, k - 1)
            h_val = self.h(x_l1)
            h_val_cat_fkm1_val = torch.cat((h_val, fkm1_val))
            # # ToDo: using fixed binary relation to see how it affects speed
            # g_val = h_val + fkm1_val
            g_val = self.g(h_val_cat_fkm1_val)
            result += g_val / len(permutations_k)
          return result

      if N_l > 1:
        x_l = X_[l, :]
        E_l = torch.cat((x_l, fk(X_[neighbors_l, :], N_l)))
      else:
        E_l = fk([X_[l, :]], N_l)
      final_l = self.final(E_l)
      final_ = torch.cat((final_, final_l))

    params = list(self.parameters())
    yhat = F.sigmoid(final_)
    return yhat

def learn_ggcn1(X_list, y_list, adjacency_list, n_epoch=100, nhid=16, batch_size=5,
               neighbor_subset_limit=2, samples_per_k=6, recursive=True, num_settings_to_try=5,
               lr=0.01, tol=0.001, dropout=0.0, neighbor_order=2):

  model = fit_ggcn1(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size,
                   neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k, 
                   recursive=recursive, lr=lr, tol=tol, dropout=dropout, neighbor_order=neighbor_order)

  def embedding_wrapper(X_): #学习得到的特征函数
    X_ = torch.FloatTensor(X_)
    E = model.embed_recursive_vec(X_).detach().numpy() 
    return E

  def model_wrapper(X_):  #最终学习出来的模型
    X_ = torch.FloatTensor(X_)
    outcome = model.forward_recursive_vec(X_, train=False) 
    yhat = outcome[:,0].detach().numpy()
    return yhat

  return embedding_wrapper, model_wrapper

def fit_ggcn1(X_list, y_list, adjacency_list, n_epoch=50, nhid=100, batch_size=5,
             neighbor_subset_limit=2, samples_per_k=6, recursive=True, lr=0.01, tol=0.001, dropout=0.0,
             locations_subsets=None, neighbor_order=2):
  # Specify model
  p = X_list[0].shape[1] #特征维数
  T = len(X_list) #拟合的总期数
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.FloatTensor(y) for y in y_list]

  model = GGCN1(nfeat=p, J=nhid, adjacency_lst=adjacency_list, neighbor_subset_limit=neighbor_subset_limit,
               samples_per_k=samples_per_k, recursive=recursive, dropout=dropout, neighbor_order=neighbor_order)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()

  
  prev_avg_acc_train = 100
  # Train
  for epoch in range(n_epoch):
    avg_acc_train = 0.
    batch_ixs = np.random.choice(T, size=batch_size)

    X_batch = [X_list[ix] for ix in batch_ixs]
    y_batch = [y_list[ix] for ix in batch_ixs]

    for X, y, ix in zip(X_batch, y_batch, batch_ixs):
      model.train()
      optimizer.zero_grad() #将梯度归零--清除上一次的梯度
      X = Variable(X)
      y = Variable(y).unsqueeze(1) #这是什么？

      if locations_subsets is not None:
        locations_subset = locations_subsets[ix]
        output = model(X, adjacency_list, locations_subset)
        loss_train = criterion(output, y[locations_subset])
      else:
        output = model(X, adjacency_list)#???????
        loss_train = criterion(output, y)
    
      regular = 0
      #L1 regularization
      LAMBDA = 0.0005
      for param in model.parameters():
        regular += torch.norm(param, 1)
      loss_train += LAMBDA*regular
      
      #print(f'regularization: {regular} loss: {loss_train}')
      loss_train.backward() 
      optimizer.step() 

    # Evaluate loss
    for X_, y_ in zip(X_list, y_list):
      yhat = model(X_, adjacency_list)[:, 0]#???????
      acc = ((yhat - y_)**2).float().mean().detach().numpy()
      avg_acc_train += acc / T

    # Break if change in accuracy is sufficiently small  这里会非常影响速度和正确率
    # if epoch > 0:
    #   relative_acc_diff = np.abs(prev_avg_acc_train - avg_acc_train) / avg_acc_train
    #   if relative_acc_diff < tol:
    #     break

    prev_avg_acc_train = avg_acc_train

  final_mse_train = 0.
  for X_, y_ in zip(X_list, y_list):
    output = model(X_, adjacency_list)
    yhat = output[:, 0] #???????
    acc = ((yhat - y_)**2).float().mean().detach().numpy()
    final_mse_train += acc / T
  print('final_mse_train: {:.4f}'.format(final_mse_train))


  return model

def oracle_tune_ggcn1(X_list, y_list, adjacency_list, env, eval_actions, true_probs,
                     X_eval=None, n_epoch=70, nhid=100, batch_size=5,
                     samples_per_k=6, recursive=True, num_settings_to_try=3,
                     X_holdout=None, y_holdout=None, neighbor_order=2):
  """
  Tune GGCN hyperparameters, given sample of true probabilities evaluated at the current state.
  """
  if env.__class__.__name__ == 'Ebola':
    NHID_RANGE = [5]
    DROPOUT_RANGE = [0.5]
    LR_RANGE = [0.008]
  else:
    NHID_RANGE = np.linspace(5, 30, 3)
    DROPOUT_RANGE = np.linspace(0, 1.0, 100)
    LR_RANGE = np.logspace(-3, -1, 100)
  NEIGHBOR_SUBSET_LIMIT_RANGE = [2]
  
  # LR_RANGE = [0.01]
  # DROPOUT_RANGE = [0.0]
  # NHID_RANGE = [16]
  

  best_predictor = None
  best_score = float('inf')
  worst_score = -float('inf')
  results = {'lr': [], 'dropout': [], 'nhid': [], 'neighbor_subset': [], 'score': []}
  for _ in range(num_settings_to_try):
    # Fit model with settings 随机选一套组合出来
    lr = np.random.choice(LR_RANGE)
    dropout = np.random.choice(DROPOUT_RANGE)
    nhid = int(np.random.choice(NHID_RANGE))
    neighbor_subset_limit = np.random.choice(NEIGHBOR_SUBSET_LIMIT_RANGE)
    
    print(f'lr: {lr} dropout: {dropout} nhid: {nhid} neighbor_limit: {neighbor_subset_limit}')

    _, predictor = learn_ggcn1(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=5,
                              neighbor_subset_limit=neighbor_subset_limit, samples_per_k=6, recursive=True, num_settings_to_try=5,
                              lr=lr, tol=0.01, dropout=dropout, neighbor_order=neighbor_order)

    # Compare to true probs
    def qfn(a):
      # X_raw_ = env.data_block_at_action(-1, a, raw=True)
      if X_eval is None:
        X_ = env.data_block_at_action(-1, a) #最后一期的
      else:
        X_ = copy.copy(X_eval)
        if hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX'):
          X_[:, 1] = a
        else:
          raise NotImplementedError
      return predictor(X_)

    phat = np.hstack([qfn(a_) for a_ in eval_actions])
    score = kl(phat, true_probs)

    if score < best_score:
      best_score = score
      best_predictor = predictor
    if score > worst_score:
      worst_score = score
    print(f'best score: {best_score} worst score: {worst_score}')
    results['lr'].append(lr)
    results['dropout'].append(dropout)
    results['nhid'].append(nhid)
    results['neighbor_subset'].append(neighbor_subset_limit)
    results['score'].append(score)

  return best_predictor, results


if __name__ == "__main__":
  # Test
  grid_size = 100
  adjacency_mat = random_nearest_neighbor(grid_size)
  adjacency_list = [[j for j in range(grid_size) if adjacency_mat[i, j]] for i in range(grid_size)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  n = 10
  n_epoch = 200
  X_list = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(n)])
  y_list = np.array([np.array([np.sum(X[adjacency_list[l]]) for l in range(grid_size)]) for X in X_list])

  # Holdout data
  X_list_holdout = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(2)])
  y_list_holdout = np.array([np.array([np.sum(X[adjacency_list[l]]) for l in range(grid_size)]) for X in X_list_holdout])

  print('Fitting ggcn')
  _, model_ = learn_ggcn1(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=16)
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
  print(f'mse: {mse} mse-l:{mse1} base:{base}')