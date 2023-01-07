import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from generate_network import lattice
from sklearn.linear_model import Ridge, LinearRegression
from functools import partial
import multiprocessing as mp
import pdb


def construct_covariance_kernel(L=300, T=25, bandwidth=1.):
  adjacency_matrix = lattice(L)
  network_as_nx_object = nx.from_numpy_matrix(adjacency_matrix)

  pairwise_distance_dictionary = dict(nx.all_pairs_shortest_path_length(network_as_nx_object))
  pairwise_distances = np.zeros((L, L))  # Entries are omega's in Nick's WNS paper
  for source_index, targets in pairwise_distance_dictionary.items():
    for target_index, length in targets.items():
      pairwise_distances[source_index, target_index] = length
      pairwise_distances[target_index, source_index] = length

  # Construct pairwise distance matrices
  pairwise_t = cdist(np.arange(T).reshape(-1, 1), np.arange(T).reshape(-1, 1))
  pairwise_t /= (np.max(pairwise_t) / bandwidth)

  pairwise_l = pairwise_distances
  pairwise_l /= (np.max(pairwise_l) / bandwidth)

  # Construct kernels
  # K_l = np.exp(-np.multiply(pairwise_l, pairwise_l)*100) # Gaussian kernel
  # K_t = np.exp(-np.multiply(pairwise_t, pairwise_t)*100)
  K_l = np.multiply(1 - pairwise_l, pairwise_l <= 1)  # Bartlett kernel
  K_t = np.multiply(1 - pairwise_t, pairwise_t <= 1)
  K = np.kron(K_t, K_l)
  return K


def generate_dependent_X_from_kernel(num_reps, K, beta):
  Xy_list = []
  n = K.shape[0]
  for _ in range(num_reps):
    # ToDo: Not right if using multivariate features
    X = np.random.multivariate_normal(mean=np.zeros(n), cov=K).reshape(-1, 1)
    y_mean = np.dot(X, beta)
    y = np.random.normal(loc=y_mean)
    Xy_list.append((X, y))
  return Xy_list


def bootstrap_regression(Xy, num_bootstrap_samples):
  X, y = Xy
  n = X.shape[0]
  reg = LinearRegression()
  coef_list = []

  for _ in range(num_bootstrap_samples):
    weights = np.random.exponential(size=n)
    reg.fit(X, y, sample_weight=weights)
    coef_list.append(reg.coef_)

  return coef_list


def distribute_boostrapping(Xy_list, num_bootstrap_samples, num_processes=48):
  pool = mp.Pool(processes=num_processes)
  bootstrap_partial = partial(bootstrap_regression, num_bootstrap_samples=num_bootstrap_samples)
  results = pool.map(bootstrap_partial, Xy_list)
  return results


def evaluate_sampling_dbns(L=300, T=25, bandwidth=1., num_bootstrap_samples=100, num_processes=48):
  BETA = np.array([1.])

  # Generate data
  K = construct_covariance_kernel(L, T, bandwidth)
  Xy_list = generate_dependent_X_from_kernel(num_bootstrap_samples, K, BETA)

  # Get bootstrap sampling dbns
  sampling_dbns = distribute_boostrapping(Xy_list, num_bootstrap_samples, num_processes=num_processes)

  # Get bootstrap coverages
  coverages = np.zeros((num_bootstrap_samples, len(BETA)))
  for b, dbn in enumerate(sampling_dbns):
    lower, upper = np.percentile(dbn, [2.5, 97.5], axis=0) # ToDo: syntax?
    for i, true_coef_i in enumerate(BETA):
      coverages[b, i] = (lower[i] < true_coef_i < upper[i])

  bootstrap_coverages = coverages.mean(axis=0)

  return bootstrap_coverages


if __name__ == "__main__":
  np.random.seed(3)
  bandwidths = [0.1, 1., 10]
  coverages_dict = {}
  for bandwidth in bandwidths:
    coverages = evaluate_sampling_dbns(L=20, T=5, bandwidth=bandwidth, num_bootstrap_samples=100, num_processes=48)
    coverages_dict[bandwidth] = coverages
  print(coverages_dict)





















