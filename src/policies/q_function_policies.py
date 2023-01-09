import os
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, '..', 'estimation/q_functions/data')


from src.estimation.q_functions.one_step import *
from src.estimation.q_functions.embedding import oracle_tune_ggcn, learn_ggcn
from src.utils.misc import kl
from sklearn.linear_model import LogisticRegression
import numpy as np



def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, raw_features = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  def oracle_qfn(a):
    return env.next_infected_probabilities(a)

  if env.learn_embedding:
    loss_dict = {}
    N_REP = 50 
    dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
    eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)]

    if env.__class__.__name__ == 'sis':
      clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
      true_probs = np.hstack([oracle_qfn(a_) for a_ in eval_actions]) 
      predictor, _ = oracle_tune_ggcn(env.X, env.y, env.adjacency_list, env, eval_actions, true_probs,
    		                      num_settings_to_try=1, n_epoch=300, neighbor_order=1) 
    else:
      clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
      def simple_oracle_qfn(a_):
        return clf.predict_proba(env.data_block_at_action(-1, a_), **predict_proba_kwargs)
      true_probs = np.hstack([simple_oracle_qfn(a_) for a_ in eval_actions])
      predictor, _ = oracle_tune_ggcn(env.X, env.y, env.adjacency_list, env, eval_actions, true_probs,
    		                      num_settings_to_try=5, n_epoch=300, neighbor_order=1)


    def qfn(a_): 
      X_ = env.data_block_at_action(-1, a_)
      return predictor(X_)

    def linear_qfn(a_): 
      return clf.predict_proba(env.data_block_at_action(-1, a_), **predict_proba_kwargs)

    def optimize_qfns(qfn_): 
      a_ = argmaxer(qfn_, evaluation_budget, treatment_budget, env)
      q_a_true_ = oracle_qfn(a_).sum()
      return a_, q_a_true_

    a, q_a_true = optimize_qfns(qfn)
    a_linear = argmaxer(linear_qfn, evaluation_budget, treatment_budget, env)
    q_alin_true = oracle_qfn(a_linear).sum()
    q_a = qfn(a).sum()
    q_alin = qfn(a_linear).sum()

    # Get accuracy at actions at _this_ timestep
    linear_acc = 0.
    gccn_acc = 0.
    linear_diffs = np.zeros(0)
    gccn_diffs = np.zeros(0)
    for a_ in eval_actions:
      linear_probs = linear_qfn(a_)
      gccn_probs = qfn(a_)
      true_probs = oracle_qfn(a_)
      gccn_acc += kl(gccn_probs, true_probs) / N_REP
      linear_acc += kl(linear_probs, true_probs) / N_REP
      linear_diffs = np.concatenate((linear_diffs, np.abs(linear_probs - true_probs)))
      gccn_diffs = np.concatenate((gccn_diffs, np.abs(gccn_probs - true_probs)))

    print(f'gccn acc: {gccn_acc} linear acc: {linear_acc}\nq(a): {q_a} q(alin): {q_alin}\nq_true(a): {q_a_true} q_alin_true: {q_alin_true}')


    q_gccn_minus_q_linear = q_a_true - q_alin_true 
    loss_dict['q_diff'] = q_gccn_minus_q_linear
  else: 
    if raw_features:
      X = np.vstack(env.X_raw)
      y = np.hstack(env.y)
      clf = LogisticRegression()
      clf.fit(X, y)
      loss_dict = {}
      def qfn(a):
        return clf.predict_proba(env.data_block_at_action(-1, a, raw=True))[:, 1]
    else:
      clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
      # Add parameters to info dictionary if params is an attribute of clf (as is the case with SKLogit2)
      if 'params' in clf.__dict__.keys():
          loss_dict['q_fn_params'] = clf.params
      def qfn(a):
        return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)
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

  # One step
  if env.learn_embedding:
    _, predictor = learn_ggcn(env.X_raw, env.y, env.adjacency_list)

    # For diagnosis
    clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)

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
      clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(1, env.T): 
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  reg = regressor()
  if raw_features:
    reg.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
  else:
    reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))

  def qfn(a):
    infections = env.Y[-1, :]
    infected_indices = np.where(infections == 1)[0]
    not_infected_indices = np.where(infections == 0)[0]
    if raw_features:
      X0 = env.data_block_at_action(-1, a, raw=True)
      X1 = env.data_block_at_action(-1, a, raw=True)
      return clf.predict_proba(X0)[:, 1] + gamma * reg.predict(X1)
    else:
      X0 = env.data_block_at_action(-1, a)
      X1 = X0
      return clf.predict_proba(X0, infected_indices, not_infected_indices) + gamma * reg.predict(X1)

      
  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, {'q_fn_params': reg.coef_}

def three_step(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['gamma'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  if env.learn_embedding:
    _, predictor = learn_ggcn(env.X, env.y, env.adjacency_list)

    # For diagnosis
    clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)

    def qfn_at_block(block_index, a):
      return predictor(env.data_block_at_action(block_index, a))
  else:
    if raw_features:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X_raw = np.vstack(env.X_raw)
      clf.fit(X_raw, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, raw=True))[:, 1]
    else:
      clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(1, env.T): 
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  reg = regressor()
  if raw_features:
    reg.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
    def qfn_at_block_1(block_index, a):
      X0 = env.data_block_at_action(block_index, a, raw = True)
      return reg.predict(X0)
  else:
    if env.learn_embedding:
      reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))
      def qfn_at_block_1(block_index, a):
        X0 = env.data_block_at_action(block_index, a)
        return reg.predict(X0)
    else:
      reg.fit(np.vstack(env.X_2[:-1]), np.hstack(backup))
      def qfn_at_block_1(block_index, a):
        X0 = env.data_block_at_action(block_index, a, neighbor_order = 2)
        return reg.predict(X0)

  #Back up 
  backup = []
  for t in range(1, env.T):
    qfn_at_block_1_t = lambda a: qfn_at_block_1(t, a)
    a_max = argmaxer(qfn_at_block_1_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_1_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  reg1 = regressor()

  if raw_features:
    reg1.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
  else:
    if env.learn_embedding:
      reg1.fit(np.vstack(env.X[:-1]), np.hstack(backup))
    else:
      reg1.fit(np.vstack(env.X_2[:-1]), np.hstack(backup))


  def qfn(a):
    infections = env.Y[-1, :]
    infected_indices = np.where(infections == 1)[0]
    not_infected_indices = np.where(infections == 0)[0]
    if raw_features:
      X0 = env.data_block_at_action(-1, a, raw=True)
      X1 = env.data_block_at_action(-1, a, raw=True)
      return clf.predict_proba(X0)[:, 1] + gamma * reg1.predict(X1)
    else:
      if env.learn_embedding:
        X0 = env.data_block_at_action(-1, a)
        X1 = X0
        return clf.predict_proba(X0, infected_indices, not_infected_indices) + gamma * reg1.predict(X1)
      else:
        X0 = env.data_block_at_action(-1, a)
        X1 = env.data_block_at_action(-1, a, neighbor_order = 2)
        return clf.predict_proba(X0, infected_indices, not_infected_indices) + gamma * reg1.predict(X1)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, {'q_fn_params': reg.coef_}

def four_step(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['gamma'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  if env.learn_embedding:
    N_REP = 50 
    dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
    eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)]

    clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
    def simple_oracle_qfn(a_):
      return clf.predict_proba(env.data_block_at_action(-1, a_), **predict_proba_kwargs)
    true_probs = np.hstack([simple_oracle_qfn(a_) for a_ in eval_actions]) 
    predictor, _ = oracle_tune_ggcn(env.X, env.y, env.adjacency_list, env, eval_actions, true_probs,
    		                  num_settings_to_try=5, n_epoch=300, neighbor_order=1) 
    
    def qfn_at_block(block_index, a):
      return predictor(env.data_block_at_action(block_index, a))
  else:
    if raw_features:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X_raw = np.vstack(env.X_raw)
      clf.fit(X_raw, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, raw=True))[:, 1]
    else:
      clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(1, env.T): 
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  reg = regressor()
  if raw_features:
    reg.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
    def qfn_at_block_1(block_index, a):
      X0 = env.data_block_at_action(block_index, a, raw = True)
      return reg.predict(X0)
  else:
    if env.learn_embedding:
      reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))
      def qfn_at_block_1(block_index, a):
        X0 = env.data_block_at_action(block_index, a)
        return reg.predict(X0)
    else:
      reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))
      def qfn_at_block_1(block_index, a):
        X0 = env.data_block_at_action(block_index, a)
        return reg.predict(X0)

  #Back up 
  backup = []
  for t in range(1, env.T):
    qfn_at_block_1_t = lambda a: qfn_at_block_1(t, a)
    a_max = argmaxer(qfn_at_block_1_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_1_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  reg1 = regressor()
  if raw_features:
    reg1.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
    def qfn_at_block_2(block_index, a):
      X0 = env.data_block_at_action(block_index, a, raw = True)
      return reg1.predict(X0)
  else:
    if env.learn_embedding:
      reg1.fit(np.vstack(env.X[:-1]), np.hstack(backup))
      def qfn_at_block_2(block_index, a):
        X0 = env.data_block_at_action(block_index, a)
        return reg1.predict(X0)
    else:
      reg1.fit(np.vstack(env.X[:-1]), np.hstack(backup))
      def qfn_at_block_2(block_index, a):
        X0 = env.data_block_at_action(block_index, a)
        return reg1.predict(X0)

  #Back up 
  backup = []
  for t in range(1, env.T):
    qfn_at_block_2_t = lambda a: qfn_at_block_2(t, a)
    a_max = argmaxer(qfn_at_block_2_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_2_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  reg2 = regressor()
  if raw_features:
    reg2.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
  else:
    if env.learn_embedding:
      reg2.fit(np.vstack(env.X[:-1]), np.hstack(backup))
    else:
      reg2.fit(np.vstack(env.X[:-1]), np.hstack(backup))
  
  def qfn(a):
    infections = env.Y[-1, :]
    infected_indices = np.where(infections == 1)[0]
    not_infected_indices = np.where(infections == 0)[0]
    if raw_features:
      X0 = env.data_block_at_action(-1, a, raw=True)
      X1 = env.data_block_at_action(-1, a, raw=True)
      return clf.predict_proba(X0)[:, 1] + gamma * reg2.predict(X1)
    else:
      if env.learn_embedding:
        X0 = env.data_block_at_action(-1, a)
        X1 = X0
        return clf.predict_proba(X0, infected_indices, not_infected_indices) + gamma * reg2.predict(X1)
      else:
        X0 = env.data_block_at_action(-1, a)
        X1 = env.data_block_at_action(-1, a)
        return clf.predict_proba(X0, infected_indices, not_infected_indices) + gamma * reg2.predict(X1)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None