import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SFIB(nn.Module):
    """
    Spatial-Frequency Integration Block (Graph-Spectral Version)
    严格遵循最终框架设计: Phase III
    """
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        
        # ==========================
        # 1. 频率域分支 (Frequency Branch - GFT)
        # ==========================
        # 接收 [N, 2C] 的频谱拼接，输出 [N, C] 的频率掩码权重
        # W_freq_mask 是可学习的参数，这里通过 MLP 动态生成
        self.freq_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid() # Mask 0~1
        )
        
        # ==========================
        # 2. 空间域分支 (Spatial Branch - Graph INO)
        # ==========================
        # 使用 GCNConv 作为 GNN_Aggregator 提取局部微环境 (S, T)
        # Guide -> Main
        self.gcn_s = GCNConv(dim, dim)
        self.gcn_t = GCNConv(dim, dim)
        
        # ==========================
        # 3. 双域融合 (Dual Fusion)
        # ==========================
        # Node Attention: 根据 diff 决定权重
        self.node_attn = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Channel Attention: 简单实现
        self.channel_fc = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim * 2),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward_freq(self, x_main, x_guide, u_basis):
        """
        频域处理: GFT -> Mask -> iGFT
        Args:
            u_basis: [N, N] 预计算的 GFT 基底
        """
        # 1. GFT (Graph Fourier Transform)
        # H_hat = U^T * H
        h_hat_main = torch.matmul(u_basis.t(), x_main)
        h_hat_guide = torch.matmul(u_basis.t(), x_guide)
        
        # 2. Spectral Masking
        # 拼接频谱
        cat_freq = torch.cat([h_hat_main, h_hat_guide], dim=1) # [N, 2C]
        
        # 生成掩码 Delta
        mask = self.freq_gate(cat_freq) # [N, C]
        
        # 残差修正: Main + Delta (这里简化为乘法门控或加法修正，按论文逻辑为加法)
        # 这里的 mask 实际上充当了 filter 的角色
        # 假设设计是: Main的频谱分布被 Guide 修正
        h_hat_fused = h_hat_main + mask * h_hat_guide 
        
        # 3. iGFT (Inverse GFT)
        # H = U * H_hat
        h_freq = torch.matmul(u_basis, h_hat_fused)
        
        return h_freq

    def forward_spatial(self, x_main, x_guide, edge_index):
        """
        空域处理: GNN Agg -> INN Coupling
        """
        # 1. Guide -> Main
        # 获取 Guide 的局部环境特征 (S, T)
        s = self.gcn_s(x_guide, edge_index)
        t = self.gcn_t(x_guide, edge_index)
        
        # 2. Affine Coupling
        # H_main' = H_main * exp(S) + T
        # 限制 S 的范围以防数值不稳定
        s = torch.tanh(s)
        h_spatial = x_main * torch.exp(s) + t
        
        return h_spatial

    def forward(self, x_main, x_guide, edge_index, u_basis):
        """
        x_main:  [N, C]
        x_guide: [N, C]
        """
        # --- Branch 1: Frequency ---
        h_freq = self.forward_freq(x_main, x_guide, u_basis)
        
        # --- Branch 2: Spatial ---
        h_spatial = self.forward_spatial(x_main, x_guide, edge_index)
        
        # --- Branch 3: Dual Interaction ---
        # 1. Node Attn (找茬)
        diff = h_spatial - h_freq
        w_node = self.node_attn(diff)
        
        # 2. 补全
        h_enhanced = h_spatial + h_freq * w_node
        
        # 3. Concat & Channel Selection
        h_cat = torch.cat([h_enhanced, h_freq], dim=1) # [N, 2C]
        
        # Global Avg/Std for Channel Attn (简化版)
        # w_channel = self.channel_fc(h_cat.mean(0, keepdim=True)) ... 省略复杂实现
        
        # 4. Final Output (Res connection handled in block usually, or here)
        out = self.out_proj(h_cat)
        
        return out + x_main # Block Residual