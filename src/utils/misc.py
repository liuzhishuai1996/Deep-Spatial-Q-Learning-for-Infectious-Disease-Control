from functools import partial
import numpy as np


def onehot(length, ix):
  arr = np.zeros(length)
  arr[ix] = 1
  return arr


def kl(p, q):
  p = np.maximum(np.minimum(p, 0.999), 0.001)
  q = np.maximum(np.minimum(q, 0.999), 0.001)
  onem_p = 1 - p
  onem_q = 1 - q
  kl_ = (p * np.log(p / q) + onem_p * np.log(onem_p / onem_q)).mean()
  return kl_


def random_argsort(arr, num_to_take):
  """
  Ad-hoc way of getting randomized argsort.
  """
  top_entries = np.sort(-arr)[:(num_to_take*2)]
  b = np.random.random(top_entries.size)
  return np.argsort(np.lexsort((b, top_entries)))[:num_to_take]


def second_order_adjacency_list(adjacency_list):
  second_order_list = []
  for l, lst in enumerate(adjacency_list):
    second_order_list_l = lst
    for lprime in lst:
      lst_lprime = adjacency_list[lprime]
      for lprimeprime in lst_lprime:
        if lprimeprime not in second_order_list_l:
          second_order_list_l = np.append(second_order_list_l, lprimeprime)
    second_order_list.append(second_order_list_l)
  return second_order_list