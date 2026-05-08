# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:25:36 2026

@author: Jianfei Li
"""

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse.csgraph import shortest_path
from sknetwork.hierarchy import Paris, LouvainHierarchy, cut_balanced
import numpy as np
import scipy.sparse as sp
import scipy.io as io
import math
import torch
import os
import pickle
import networkx as nx

"""
Example Script: 
Verifies the Tight Frame property on random graphs using 
Fast Graph Framelet Transform and Explicit Matrix Representation.
"""

def get_spatial_partitions(adj, h = 4):
    
    # print("=======Generating hierachical partitions======")

    n = adj.shape[0]
    cluster_size_bound = h
    
    paris = Paris()
    partitions= []
    partitions.append([i for i in range(n)])
    prev_cluster_set = {}
    for i in range(n):
        temp_set = set()
        temp_set.add(i)
        prev_cluster_set[i] = temp_set
    prev_adj = adj
    while(prev_adj.shape[0]>cluster_size_bound and np.abs(np.sum(prev_adj))>1e-6):
        # print("Current graph size：",prev_adj.shape)
        dendrogram = paris.fit_transform(prev_adj)
        cluster_id = cut_balanced(dendrogram, cluster_size_bound)
        cluster_num = max(cluster_id) + 1
        
        temp_cluster_set = {}
        temp_cluster_list = {}
        for j in range(len(cluster_id)):
            temp_id = cluster_id[j]
            if temp_id not in temp_cluster_set.keys():
                temp_cluster_set[temp_id] = prev_cluster_set[j]
                temp_cluster_list[temp_id] = [j]
            else:
                temp_cluster_set[temp_id] = temp_cluster_set[temp_id].union(prev_cluster_set[j])
                temp_cluster_list[temp_id].append(j)
        
        prev_cluster_set = temp_cluster_set
        temp_adj = np.zeros((cluster_num,cluster_num))
        for j in range(cluster_num):
            for k in range(j+1,cluster_num):
                edge_weight = 0.0
                for p in temp_cluster_list[j]:
                    for q in temp_cluster_list[k]:
                        edge_weight += prev_adj[p][q]
                temp_adj[j][k]=temp_adj[k][j] = edge_weight
        prev_adj = temp_adj
        
        temp_partition = [0 for j in range(n)]
        for j in range(cluster_num):
            temp_list = list(temp_cluster_set[j])
            for p in temp_list:
                temp_partition[p] = j
        # print("Current level cluster num:",max(temp_partition)+1)
        
        partitions.append(temp_partition)
    
    # print("=======Generation completed=======")
    partitions.append([0 for i in range(n)])
    return partitions

def generate_random_binary_adj(n=20, p=0.2):
    """
    生成一个具有 n 个节点、连边概率为 p 的随机 0-1 邻接矩阵
    """
    # 生成随机图 G(n, p)
    G = nx.erdos_renyi_graph(n, p, seed=42)
    # 转换为 numpy 邻接矩阵 (只有 0 和 1)
    adj = nx.to_numpy_array(G, weight=None)
    return adj


# 1. Generate a random graph with 50 nodes.
n_nodes = 50
adj = generate_random_binary_adj(n=n_nodes, p=0.15)

# 2. get a partition list
partitions_list = get_spatial_partitions(adj, h=4)
partition_matrix = torch.tensor(np.array(partitions_list), dtype=torch.long)

print("\n" + "="*60)
print(f"Success: Generated Partition Matrix with shape {partition_matrix.shape}")
print(f"Nodes: {n_nodes}, Hierarchical Levels: {partition_matrix.shape[0]}")
print("="*60)

# 3. test fast graph framelet transform and graph framelet matrix
from binary_haar_graph_framelets import GraphFrameletTransform

device = "cuda"
target_level = partition_matrix.shape[0] - 2
signal = torch.rand(n_nodes, 10) # feature dimension =10

transform = GraphFrameletTransform(partition_matrix)


coeffs, scale_indices = transform.fft(signal, level=target_level)
print(">>> Fast framelet transform")
print(f"Decomposition Level: {target_level}")
print(f"Coefficient Matrix Shape: {list(coeffs.shape)}")
print(f"Scale Indices: {scale_indices.long().tolist()}")
print("-"*30)

# 
energy_error = torch.abs(torch.norm(signal) - torch.norm(coeffs))
print(f"Energy Preservation Error (Parseval): {energy_error.item():.2e}")

# 
signal_rec = transform.ifft(coeffs, scale_indices)
recon_error = torch.norm(signal - signal_rec)
print(f"Total Reconstruction Error (L2):   {recon_error.item():.2e}")


# fft v.s. matrix

print(">>> Generating and Verifying Graph Framelet Matrix")
print("-"*60)


W, m_scale_indices = transform.compute_framelet_matrix(level=target_level)

# y = W * x
coeffs_matrix = torch.matmul(W, signal)

# x_hat = W^T * y
signal_rec_matrix = torch.matmul(W.t(), coeffs_matrix)

# ifft "=" W.t()
recon_consistency = torch.norm(signal_rec_matrix - signal_rec)
print(f"Matrix Recon vs FFT Recon: {recon_consistency.item():.2e}")


coeffs_consistency = torch.norm(coeffs_matrix - coeffs)
print(f"Matrix Coeffs vs FFT Coeffs: {coeffs_consistency.item():.2e}")


