import numpy as np
import networkx as nx
import pdb

'''
Functions for generating adjacency matrices for networks to be used in SpatialDisease
sims.
'''


def pseudo_square_root(integer):
  """
  求integer的伪平方根
  Stupid search for pseudo square root (largest factor that doesn't exceed sqrt(integer).)
  """
  ###### 本来带了括号 我给去掉了 ######
  assert integer < 1e6, "Number too big, choose something less than 1e6." 
  sqrt = np.sqrt(integer)
  psr = 1
  psr_complement = integer
  i = 2
  while i <= sqrt:
    if integer % i == 0:
      psr = i
      psr_complement = integer / i
    i += 1
  return psr, psr_complement


def lattice(size):
  """
  生成网格状network的邻接矩阵，格子的长宽分别是size伪平方根对应的两个因子
  Return adjacency matrix for sqrt(size) x sqrt(size) lattice.
  :param size:
  :return:
  """
  nrow, ncol = pseudo_square_root(size)
  adjacency_matrix = np.zeros((size, size))
  for i in range(size):
    for j in range(size):
      if (j == i + 1) and ((i + 1) % ncol != 0):
        adjacency_matrix[i, j] = 1
      elif (j == i - 1) and (i % ncol != 0):
        adjacency_matrix[i, j] = 1
      elif (j == i + ncol) and (i + 1 + nrow <= size):
        adjacency_matrix[i, j] = 1
      elif (j == i - ncol) and (i + 1 - nrow > 0):
        adjacency_matrix[i, j] = 1
  return adjacency_matrix


def update_adjacency_matrix(adjacency_matrix, ix, neighbors_lst):
  adjacency_matrix[ix, neighbors_lst] = 1
  return adjacency_matrix


def contrived(size):
  '''
  人工生成的7个单位为一个子网的网络
  '''
  num_subnets = size // 7
  new_size = num_subnets * 7
  adjacency_matrix = np.zeros((new_size, new_size))
  for subnet_ix in range(num_subnets):
    num_so_far = subnet_ix * 7

    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far, [num_so_far + 1, num_so_far + 2])
    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far + 1, [num_so_far, num_so_far + 2])
    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far + 2,
                                               [num_so_far, num_so_far + 1 , num_so_far + 3])
    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far + 3, [num_so_far + 2, num_so_far + 4])
    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far + 4, [num_so_far + 3, num_so_far + 5,
                                                                                  num_so_far + 6])
    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far + 5, [num_so_far + 4, num_so_far + 6])
    adjacency_matrix = update_adjacency_matrix(adjacency_matrix, num_so_far + 6, [num_so_far + 4, num_so_far + 5])

  return adjacency_matrix


def contrived_initial_infections(size):
  #初始感染 每一个子网中第一个和第七个点
  num_subnets = size // 7
  new_size = num_subnets * 7
  infections = np.zeros(new_size)
  for subnet_ix in range(num_subnets):
    num_so_far = subnet_ix * 7
    infections[num_so_far] = 1
    infections[num_so_far + 6] = 1
  return infections


def Barabasi_Albert(size, prop_init=0.9, p=0.05):
  """
  生成子群网络 网络中部分节点自由度特别高
  Random preferential attachment model
  See https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model#Algorithm
  """
  PROP_INIT = prop_init  # 0.2  # Initial graph size will be PROP_INIT * size
  P = p  # 0.5  # Probability of uniform random connection (otherwise, preferentially attach)

  # Initialize graph
  initial_graph_size = int(np.floor(PROP_INIT * size))
  assert initial_graph_size > 0

  # Start with fully connected adjacency matrix
  adjacency_matrix = np.zeros((size, size))
  for l in range(initial_graph_size):
    for l_prime in range(initial_graph_size):
      adjacency_matrix[l, l_prime] = 1

  for l in range(initial_graph_size, size):
    # Randomly attach
    if np.random.random() < P:
      l_prime = np.random.choice(l)
    else:
      degrees = np.sum(adjacency_matrix[:l, ], axis=1)
      probs = degrees / np.sum(degrees)
      l_prime = np.random.choice(l, p=probs)
    adjacency_matrix[l, l_prime] = 1
    adjacency_matrix[l_prime, l] = 1

  # Make sure every location has at least one neighbor
  for l in range(size):
    if np.sum(adjacency_matrix[l, :]) == 1:
      lprime = np.random.choice(size)
      adjacency_matrix[l, lprime] = adjacency_matrix[lprime, l] = 1

  return adjacency_matrix


def order_stat(arr, k):
  """
  np.partition(arr, k):先对arr从小到大排序，以第k大的数为基准，调整其他的数。小于它的放到它左边，大于它的放到它右边
  这个代码可以找到从小到大排第k的数(从1起)
  """
  return np.partition(arr, k)[:k].max()

def argmin_2d(arr):
  '''
  argmin():最小元素的位置
  unravel_index(indices, shpe): 原来向量中indices位置的数在转成矩阵以后的位置
  '''
  return np.unravel_index(arr.argmin(), arr.shape)


def random_nearest_neighbor(size):

  # Described in RSS-Supplement
  k = 3
  C = np.random.random(size=(size, 2))
  E = np.array([
    np.array([np.linalg.norm(C[i, :] - C[j, :]) for j in range(size)])
    for i in range(size)])
  N_k = [[lprime for lprime in range(size) if E[l, lprime] <= np.sort(E[l, :])[k]]
         for l in range(size)]
  omega_tilde = np.array([
    np.array([lprime in N_k[l] for lprime in range(size)])
    for l in range(size)])

  omega_tilde = nx.from_numpy_matrix(omega_tilde)
  B = list(nx.connected_components(omega_tilde))
  while len(B) > 1:
    best_pair = None
    best_dist = float('inf')
    for i in range(len(B)):
      for j in range(i + 1, len(B)):
        B_i = [x for x in B[i]]
        B_j = [x for x in B[j]]
        distances = np.array([[E[l, lprime] for l in B_i]
                              for lprime in B_j])
        min_dist = np.min(distances)
        if best_pair is None or min_dist < best_dist:
          best_pair_ixs = argmin_2d(distances)
          best_pair = B_i[best_pair_ixs[1]], B_j[best_pair_ixs[0]]
          best_dist = min_dist
    omega_tilde.add_edge(best_pair[0], best_pair[1])
    B = list(nx.connected_components(omega_tilde))
  TEMP = np.array(nx.adjacency_matrix(omega_tilde).todense())
  for i in range(size):
    TEMP[i][i] = 0
  return TEMP


if __name__ == "__main__":
  for i in range(100):
    print(i)
    m = random_nearest_neighbor(100)

