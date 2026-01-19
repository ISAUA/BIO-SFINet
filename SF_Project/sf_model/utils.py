import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def build_spatial_graph(coords, k=10):
    """
    构建空间 KNN 图并计算 GFT 基底。
    对应框架: Phase 1 - Global Graph Basis Construction
    
    Args:
        coords: [N, 2] 原始物理坐标 (numpy array)
        k: 近邻数
    Returns:
        edge_index: [2, E] 图的边索引 (供 GAT/GNN 使用)
        u_basis: [N, N] 拉普拉斯矩阵的特征向量矩阵 (供 GFT 使用)
    """
    N = coords.shape[0]
    
    # 1. 构建 KNN 图
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # 构建边索引 (Edge Index)
    # indices 第一列是自己，从第二列开始是邻居
    src = np.repeat(np.arange(N), k)
    dst = indices[:, 1:].flatten()
    
    # 转为 PyG 格式的 edge_index [2, E]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 2. 构建归一化拉普拉斯矩阵 L
    # 构造稀疏邻接矩阵
    data = np.ones(len(src))
    adj = sp.coo_matrix((data, (src, dst)), shape=(N, N))
    
    # 对称化 (变为无向图)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # 计算度矩阵 D
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    laplacian = sp.eye(N) - normalized_adj
    
    # 3. 特征分解 (Eigen Decomposition)
    # L = U * Lambda * U^T
    # 对于 N=2500，dense solver (eigh) 速度很快
    evals, evecs = np.linalg.eigh(laplacian.toarray())
    
    # 排序 (低频 -> 高频)
    idx = np.argsort(evals)
    # evals = evals[idx]
    evecs = evecs[:, idx]
    
    # 转为 Tensor
    u_basis = torch.FloatTensor(evecs)
    
    return edge_index, u_basis

class CLIPLoss(nn.Module):
    """
    Phase 4: Contrastive Alignment Loss
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_rna, z_atac):
        # L2 Normalize
        z_rna = F.normalize(z_rna, dim=1)
        z_atac = F.normalize(z_atac, dim=1)
        
        # Similarity Matrix
        logits = torch.matmul(z_rna, z_atac.T) / self.temperature
        
        # Labels: Diagonal is positive pair
        batch_size = z_rna.shape[0]
        labels = torch.arange(batch_size).to(z_rna.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2