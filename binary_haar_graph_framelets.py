# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:25:36 2026

@author: Jianfei Li
"""

import torch
import math


def get_difference_matrix(d):
    """Construct difference matrix A for d sub-clusters [Corollary 1]."""
    if d == 1:
        return torch.tensor([[1.0]])
    
    n_rows = int(d * (d - 1) / 2)
    A = torch.zeros((n_rows, d))
    
    row = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[row, i] = 1.0
            A[row, j] = -1.0
            row += 1
    return A

def normalize_tight_frame(A):
    """Normalize A to ensure tight frame property [Theorem 1]."""
    # 计算归一化常数，确保 AA^T 的性质
    C = torch.matmul(A, A.t())
    D = torch.matmul(C, A)
    factor = torch.sqrt(D[0, 0])
    return A / factor

def get_inverse_matrix(d):
    """Connstruct matrix C for fast graph framelet transform [Theorem 5] """
    A = get_difference_matrix(d)
    A = normalize_tight_frame(A)
    
    p = torch.ones((1,d)) / math.sqrt(d)
    
    P = torch.vstack((p,A))
    C = P.T
    
    return C



class GraphFrameletTransform:
    def __init__(self, partition_matrix, device = "cpu"):
        """
        Args:
            partition_matrix (Tensor): (Levels, Nodes) hierarchical clustering labels.
            device (str): 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.partition_matrix = partition_matrix
        self.num_levels, self.n = partition_matrix.shape

    
    def fft(self, signal, level):
        """
        fast_framelet_transform: 基于分区矩阵计算完整的小波变换矩阵 W[cite: 6, 8]。
        """
        high = []
        scales = []
        low_current = signal.to(self.device)
        
        for j in range(1,level+1):
            current_labels = self.partition_matrix[j]
            unique_clusters = torch.unique(current_labels)
            
            low_next_level = []
            
            for cluster_id in unique_clusters:
                vertex_indices = (current_labels == cluster_id).nonzero(as_tuple=True)[0]
                #tracks the indices of the sub-clusters from the previous level
                indices = torch.unique(self.partition_matrix[j-1][vertex_indices])
                
                ell = len(indices)
                
                if ell > 1:
                    C = get_inverse_matrix(ell).to(self.device)
                    f = torch.matmul(C.T, low_current[indices,:]).to(self.device)
                    # Low-pass component
                    low_next_level.append(f[0,:])
                    # High-pass components
                    high.append(f[1::,:])
                    # indices of scales
                    scales.append(j*torch.ones(f.shape[0]-1))
            
            # update low-pass components for next level
            low_current = torch.vstack(low_next_level)
            
        scale_indices = torch.cat([j*torch.zeros(low_current.shape[0])] + scales, dim=0)
        coefficients = torch.vstack([low_current] + high)
        return coefficients, scale_indices
    
    
    def ifft(self, coefficients, scale_indices):
        """
        fast_framelet_transform: 基于分区矩阵计算完整的小波变换矩阵 W[cite: 6, 8]。
        """
        current_level = int(torch.max(scale_indices).item())
        # low-pass coeffs
        num_low = (scale_indices == 0).sum().item()
        low_current = coefficients[:num_low]
        
        # Coarse-to-Fine
        for j in range(current_level,0,-1):
            current_labels = self.partition_matrix[j]
            unique_clusters = torch.unique(current_labels)
            
            num_clusters_prev = len(torch.unique(self.partition_matrix[j-1]))
            low_next_level = torch.zeros(num_clusters_prev,coefficients.shape[1],device=self.device)
            
            high_current = coefficients[scale_indices==j,:]
            
            for cluster_id in unique_clusters:
                vertex_indices = (current_labels == cluster_id).nonzero(as_tuple=True)[0]
                # tracks the indices of the sub-clusters
                indices = torch.unique(self.partition_matrix[j-1][vertex_indices])
                
                ell = len(indices)
                
                if ell > 1:
                    p = torch.ones((ell,1), device=self.device) / math.sqrt(ell)
                    A = get_difference_matrix(ell).to(self.device)
                    A = normalize_tight_frame(A)
                    num_high = A.shape[0] # number of high-pass coeffs
                    # reconstruction
                    low_next_level[indices] = torch.matmul(p, low_current[cluster_id:cluster_id+1])+ torch.matmul(A.t(), high_current[:num_high,:])
                    
                    high_current = high_current[num_high::,:] # exclude used high-pass coeffs
                
            low_current = low_next_level
        
        recon_signal = low_current
        
        return recon_signal
    
    def compute_framelet_matrix(self, level):
        """
        Generate graph framelet matrix
        """
        phi_current = torch.eye(self.n)
        all_psi = []
        scales = []
        
        for j in range(1,level+1):
            current_labels = self.partition_matrix[j]
            unique_clusters = torch.unique(current_labels)
            phi_next_level = []
            
            for cluster_id in unique_clusters:
                vertex_indices = (current_labels == cluster_id).nonzero(as_tuple=True)[0]
                indices = torch.unique(self.partition_matrix[j-1][vertex_indices])
                sub_phi = phi_current[indices]
                
                ell = len(indices)
                
                if ell > 1:
                    phi0 = torch.sum(sub_phi / math.sqrt(ell), dim=0)
                    phi_next_level.append(phi0)
                    
                    A = get_difference_matrix(ell)
                    A = normalize_tight_frame(A)
                    psi = torch.matmul(A, sub_phi)
                    all_psi.append(psi)
                    
                    scales.append(j*torch.ones(psi.shape[0]))
                    
                else:
                    phi_next_level.append(sub_phi[0])
                    
            phi_current = torch.stack(phi_next_level)
            
        scale_indices = torch.cat([torch.zeros(phi_current.shape[0])] + scales, dim=0)
        W = torch.cat([phi_current] + all_psi, dim=0).to(self.device)
        return W, scale_indices
    

if __name__ == "__main__":
        
    device = "cuda"
    # --- 1. 验证数学基础 (定理证明) ---
    print("="*60)
    print(">>> Phase 1: Verifying Mathematical Foundations (Theorem 1 & 5)")
    print("-"*60)
    for ell in range(2, 10):
        A = get_difference_matrix(ell)
        A = normalize_tight_frame(A).to(device)
        
        p = torch.ones((1, ell), device=device) / math.sqrt(ell)
        P = torch.vstack((p, A))
        C = get_inverse_matrix(ell).to(device)
        
        diff = torch.abs(torch.matmul(C, P) - torch.eye(ell, device=device)).sum()
        print(f"Dimension d={ell:2d} | Residual Error: {diff.item():.2e}")
    print("\n")

    # --- 2. 验证快速变换算法 ---
    print("="*60)
    print(">>> Phase 2: Evaluating Fast Graph Framelet Transform")
    print("-"*60)
    
    partition_data = torch.tensor([
        [0,1,2,3,4,5,6,7,8,9],
        [0,1,0,2,3,0,1,2,3,3],
        [0,0,0,1,1,0,0,1,1,1], 
        [0,0,0,0,0,0,0,0,0,0]
    ])
    
    signal = torch.rand(10, 15, device=device)
    transform = GraphFrameletTransform(partition_data, device=device)
    
    target_level = 3
    coeffs, scale_indices = transform.fft(signal, level=target_level)
    
    print(f"Decomposition Level: {target_level}")
    print(f"Coefficient Matrix Shape: {list(coeffs.shape)}")
    print(f"Scale Indices: {scale_indices.long().tolist()}")
    print("-"*30)

    # 
    energy_error = torch.abs(torch.norm(signal) - torch.norm(coeffs))
    print(f"Energy Preservation Error (Parseval): {energy_error.item():.2e}")
    
    if energy_error < 1e-5:
        print("[Status] Tight Frame Property: VERIFIED")
    
    # 
    signal_rec = transform.ifft(coeffs, scale_indices)
    recon_error = torch.norm(signal - signal_rec)
    print(f"Total Reconstruction Error (L2):   {recon_error.item():.2e}")
    
    if recon_error < 1e-5:
        print("[Status] Perfect Reconstruction:    SUCCESS")
    print("="*60)
    
    
    # --- 3. 验证显式框架矩阵 (矩阵形式 vs 快速算法) ---
    print("="*60)
    print(">>> Phase 3: Generating and Verifying Graph Framelet Matrix")
    print("-"*60)
    
    
    W, m_scale_indices = transform.compute_framelet_matrix(level=target_level)
    
    # y = W * x
    coeffs_matrix = torch.matmul(W, signal)
    
    # x_hat = W^T * y
    signal_rec_matrix = torch.matmul(W.t(), coeffs_matrix)
    
    # ifft "=" W.t()
    recon_consistency = torch.norm(signal_rec_matrix - signal_rec)
    print(f"Consistency (Matrix Recon vs FFT Recon): {recon_consistency.item():.2e}")
    
    
    coeffs_consistency = torch.norm(coeffs_matrix - coeffs)
    print(f"Consistency (Matrix Coeffs vs FFT Coeffs): {coeffs_consistency.item():.2e}")

    # W^T * W = I
    I_approx = torch.matmul(W.t(), W)
    identity_error = torch.norm(I_approx - torch.eye(signal.shape[0], device=device))
    print(f"Tight Frame Identity Error (W'W - I):    {identity_error.item():.2e}")
    
    if recon_consistency < 1e-5 and coeffs_consistency < 1e-5:
        print("[Status] Matrix Representation:          VERIFIED & ALIGNED")
    
    if identity_error < 1e-5:
        print("[Status] Tightness:        PERFECT")
        
    print("="*60)