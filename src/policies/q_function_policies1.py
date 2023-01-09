import os
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, '..', 'estimation/q_functions/data')


from src.estimation.q_functions.one_step import *
from src.estimation.q_functions.embedding import oracle_tune_ggcn, learn_ggcn
from src.estimation.q_functions.embedding1 import oracle_tune_ggcn1, learn_ggcn1
from src.utils.misc import kl
from sklearn.linear_model import LogisticRegression
import numpy as np



def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, raw_features = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L) #从权重生成的数量来看 应该就是我写的那样
  else:
    weights = None
  loss_dict = {}
  if env.learn_embedding: 
    _, predictor = learn_ggcn(env.X_raw, env.y, env.adjacency_list, neighbor_order=1)
    def qfn(a_):
      X_ = env.data_block_at_action(-1, a_, raw = True)
      return predictor(X_)
  else:
    if raw_features:
      X = np.vstack(env.X_raw)
      y = np.hstack(env.y)
      clf = LogisticRegression()
      clf.fit(X, y)
      def qfn(a_):
        return clf.predict_proba(env.data_block_at_action(-1, a_, raw=True))[:, 1]
    else:
      X = np.vstack(env.X_2)
      y = np.hstack(env.y)
      clf = LogisticRegression()
      clf.fit(X, y)
      def qfn(a_):
        return clf.predict_proba(env.data_block_at_action(-1, a_, neighbor_order=2))[:, 1]
      
  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)

  return a, loss_dict

def two_step(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['gamma'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None
  loss_dict = {}
  # One step
  if env.learn_embedding:
    _, predictor = learn_ggcn(env.X_raw, env.y, env.adjacency_list)
    # For diagnosis
    clf = LogisticRegression()
    y = np.hstack(env.y)
    X = np.vstack(env.X)
    clf.fit(X, y)

    def qfn_at_block(block_index, a):
      return predictor(env.data_block_at_action(block_index, a, raw=True))
  else:
    if raw_features:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X_raw = np.vstack(env.X_raw)
      clf.fit(X_raw, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, raw=True))[:, 1]
    else:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X = np.vstack(env.X_2)
      clf.fit(X, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, neighbor_order=2))[:, 1]

  # Back up once
  backup = []
  for t in range(1, env.T): 
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  if env.learn_embedding:
    y = []
    for i in range(len(backup)):
      y.append(backup[i])
    _, predictor1 = learn_ggcn1(env.X_raw[:-1], y, env.adjacency_list, neighbor_order=1)
    def qfn(a_):
      #infections = env.Y[-1, :]
      #infected_indices = np.where(infections == 1)[0]
      #not_infected_indices = np.where(infections == 0)[0]
      X0 = env.data_block_at_action(-1, a_)
      X_ = env.data_block_at_action(-1, a_, raw = True)
      return clf.predict_proba(X0)[:,1] + gamma*predictor1(X_)
  else:
    reg = regressor()
    if raw_features:
      y = np.hstack(env.y[:-1])
      reg.fit(np.vstack(env.X_raw[:-1]), y + gamma*np.hstack(backup))
      def qfn(a_):
        return reg.predict(env.data_block_at_action(-1, a_, raw = True))
    else:
      y = np.hstack(env.y[:-1])
      reg.fit(np.vstack(env.X_2[:-1]), y + gamma*np.hstack(backup))
      def qfn(a_):
        return reg.predict(env.data_block_at_action(-1, a_, neighbor_order = 2))
    
  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, loss_dict

def three_step(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['gamma'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None
  loss_dict = {}

  if env.learn_embedding:
    _, predictor = learn_ggcn(env.X_raw, env.y, env.adjacency_list)

    clf = LogisticRegression()
    y = np.hstack(env.y)
    X = np.vstack(env.X)
    clf.fit(X, y)

    def qfn_at_block(block_index, a):
      return predictor(env.data_block_at_action(block_index, a, raw = True))
  else:
    if raw_features:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X = np.vstack(env.X_raw)
      clf.fit(X, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, raw=True))[:, 1]
    else:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X = np.vstack(env.X_2)
      clf.fit(X, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, neighbor_order=2))[:, 1]

  # Back up once
  backup = []
  for t in range(1, env.T): 
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  if env.learn_embedding:
    y = []
    for i in range(len(backup)):
      y.append(backup[i])
    _, predictor1 = learn_ggcn1(env.X_raw[:-1], y, env.adjacency_list)
    def qfn_at_block_1(block_index, a):
      X0 = env.data_block_at_action(block_index, a)
      X_ = env.data_block_at_action(block_index, a, raw=True)
      return clf.predict_proba(X0)[:,1]+ gamma*predictor1(X_) 
  else:
    reg = regressor()
    if raw_features:
      y = np.hstack(env.y[:-1])
      reg.fit(np.vstack(env.X_raw[:-1]), y + gamma * np.hstack(backup))
      def qfn_at_block_1(block_index, a):
        return reg.predict(env.data_block_at_action(block_index, a, raw = True))
    else:
      y = np.hstack(env.y[:-1])
      reg.fit(np.vstack(env.X_2[:-1]), y + gamma * np.hstack(backup))
      def qfn_at_block_1(block_index, a):
        return reg.predict(env.data_block_at_action(block_index, a, neighbor_order=2))

  #Back up 
  backup1 = []
  for t in range(1, env.T):
    qfn_at_block_1_t = lambda a: qfn_at_block_1(t, a)
    a_max = argmaxer(qfn_at_block_1_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_1_t(a_max)
    backup1_at_t = q_max
    backup1.append(backup1_at_t)
  
  #The third step
  
  if env.learn_embedding:
    y = []
    for i in range(len(backup)):
      y.append(backup1[i])
    _, predictor2 = learn_ggcn1(env.X_raw[:-1], y, env.adjacency_list)
    def qfn(a_):
      X0 = env.data_block_at_action(-1, a_)
      X_ = env.data_block_at_action(-1, a_, raw = True)
      return clf.predict_proba(X0)[:,1] + gamma*predictor2(X_)
  else:
    reg1 = regressor()
    if raw_features:
      y = np.hstack(env.y[:-1])
      reg1.fit(np.vstack(env.X_raw[:-1]), y + gamma * np.hstack(backup1))
      def qfn(a_):
        return reg1.predict(env.data_block_at_action(-1, a_, raw = True))
    else:
      y = np.hstack(env.y[:-1])
      reg1.fit(np.vstack(env.X_2[:-1]), y + gamma * np.hstack(backup1))
      def qfn(a_):
        return reg1.predict(env.data_block_at_action(-1, a_, neighbor_order=2))

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, loss_dict

