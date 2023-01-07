import numpy as np


def fit_one_step_predictor(classifier, env, weights, truncate=False, y_next=None, print_compare_with_true_probs=True,
                           indices=None):
  clf = classifier()
  if indices is None:
    if y_next is None:
      target = np.hstack(env.y).astype(float)
    else:
      target = y_next
    features = np.vstack(env.X) 
  else:
    target = np.hstack([y_[ixs_at_t] for ixs_at_t, y_ in zip(indices, env.y)])
    features = np.vstack([x[ixs_at_t, :] for ixs_at_t, x in zip(indices, env.X)])

  if clf.condition_on_infection:
    if indices is None:
      X_raw = np.vstack(env.X_raw)
    else:
      X_raw = np.vstack([x_raw[ixs_at_t, :] for ixs_at_t, x_raw in zip(indices, env.X_raw)])
    clf_kwargs = {'infected_locations': np.where(X_raw[:, -1] == 1),
                  'not_infected_locations': np.where(X_raw[:, -1] == 0)}
    predict_proba_kwargs = {'infected_locations': np.where(env.X_raw[-1][:, -1] == 1)[0],
                            'not_infected_locations': np.where(env.X_raw[-1][:, -1] == 0)[0]}
  else:
    clf_kwargs = {}
    predict_proba_kwargs = {}

  if weights is not None:
    weights = weights.flatten()

  clf.fit(features, target, weights, truncate, **clf_kwargs)

  info = {}
  return clf, predict_proba_kwargs, info
