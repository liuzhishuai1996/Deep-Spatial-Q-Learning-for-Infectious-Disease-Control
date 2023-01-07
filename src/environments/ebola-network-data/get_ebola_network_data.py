# -*- coding: utf-8 -*-
"""
Created on Sun May 20 18:25:23 2018

@author: Jesse
"""

import numpy as np
import pandas as pd
import pickle as pkl
from math import radians, cos, sin, asin, sqrt
import sys
import os
import pdb

this_fpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_fpath)

L = 290 #290 locations in Gravity sim

def haversine(lon1, lat1, lon2, lat2):
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
  c = 2*asin(sqrt(a))
  r = 6371
  return c * r

def main():
  
  ##Create adjacency matrix
  edges = pd.read_table('ebola_edges.txt', header=None)
  adjacency_matrix = np.zeros((L, L))
  edges_list = edges[0]
  
  #Loop through each edge pair, parse, add to adj matrix
  for edge_pair_str in edges_list:
    edge_pair = list(map(int, edge_pair_str.split(' ')))
    edge_0 = edge_pair[0]
    edge_1 = edge_pair[1]
    adjacency_matrix[edge_0, edge_1] = 1

  ##Create distance matrix
  haversine_distance_matrix = np.zeros((L, L))
  euclidean_distance_matrix = np.zeros((L, L))

  def str_to_float(coords_text_file):
    f = open(coords_text_file, 'r')
    f_str = f.read().split('\n')
    coords = list(map(float, f_str[:-1]))
    return coords

  x_coords = str_to_float('ebola_x.txt')
  y_coords = str_to_float('ebola_y.txt')
  
  #Get pairwise distances for every adjacent pair
  for i in range(L):
    for j in range(L):
      lon1, lat1 = x_coords[i], y_coords[i]
      lon2, lat2 = x_coords[j], y_coords[j]
      haversine_distance_matrix[i, j] = haversine(lon1, lat1, lon2, lat2)
      euclidean_distance_matrix[i, j] = np.linalg.norm(np.array([lon1 - lon2, lat1 - lat2]))
  ##Get populations
  pop_list = str_to_float('ebola_population.txt')
  pop_array = np.array(pop_list)

  outbreak_time_list = str_to_float('ebola_outbreaks.txt')
  outbreak_time_array = np.array(outbreak_time_list).astype(int)  

  ebola_network_data = {'adjacency_matrix':adjacency_matrix, 'haversine_distance_matrix':haversine_distance_matrix, 
                        'euclidean_distance_matrix':euclidean_distance_matrix, 'pop_array':pop_array,
                        'outbreak_time_array':outbreak_time_array}
  pkl.dump(ebola_network_data, open('ebola_network_data.p', 'wb'), protocol=2)  #Need protocol < 3 for P2.7 compatibility

  return
 
if __name__ == '__main__':
  main()
