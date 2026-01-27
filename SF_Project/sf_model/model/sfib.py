import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class INOUnit(nn.Module):
    """双向仿射耦合的 INO 单元，Guide⇄Main 互修。"""
    def __init__(self, dim: int):
        super().__init__()
        # Guide 修 Main
        self.gcn_s1 = GCNConv(dim, dim)
        self.gcn_t1 = GCNConv(dim, dim)
        # Main* 反向修 Guide
        self.gcn_s2 = GCNConv(dim, dim)
        self.gcn_t2 = GCNConv(dim, dim)

    def forward(self, x_main, x_guide, edge_index):
        # Step1: Guide -> Main
        s1 = torch.tanh(self.gcn_s1(x_guide, edge_index))
        t1 = self.gcn_t1(x_guide, edge_index)
        x_main_star = x_main * torch.exp(s1) + t1

        # Step2: Main* -> Guide
        s2 = torch.tanh(self.gcn_s2(x_main_star, edge_index))
        t2 = self.gcn_t2(x_main_star, edge_index)
        x_guide_star = x_guide * torch.exp(s2) + t2

        return x_main_star, x_guide_star

class SFIB(nn.Module):
    """
    Spatial-Frequency Integration Block (Graph-Spectral Version)
    严格遵循最终框架设计: Phase III
    """
    def __init__(self, dim=128, num_ino_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.num_ino_layers = int(num_ino_layers)
        
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
        # 堆叠可配置数量的双向 INO 单元
        self.ino_layers = nn.ModuleList([INOUnit(dim) for _ in range(self.num_ino_layers)])
        # 拼接 Main/Guide 后用 1x1 线性压回 dim 维
        self.spa_proj = nn.Linear(dim * 2, dim)
        
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
        # 细节注入强度参数 gamma
        self.gamma = nn.Parameter(torch.tensor(1.0))

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
        空域处理: 3 层 INO 级联双向互修 + 拼接压缩
        """
        main, guide = x_main, x_guide
        for ino in self.ino_layers:
            main, guide = ino(main, guide, edge_index)
        h_spatial = self.spa_proj(torch.cat([main, guide], dim=1))
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
        # A. 提取高频残差 (反锐化掩模)
        f_detail = h_spatial - h_freq

        # B. 自适应细节注入权重
        w_inject = self.node_attn(f_detail)  # 使用已有 MLP+Sigmoid
        f_sharp = w_inject * f_detail

        # C. 频域为基底，加权细节叠加
        h_enhanced = h_freq + self.gamma * f_sharp

        # 拼接：保持与增强后的基底对应
        h_cat = torch.cat([h_enhanced, h_spatial], dim=1) # [N, 2C]
        
        # Global Avg/Std for Channel Attn (简化版)
        # w_channel = self.channel_fc(h_cat.mean(0, keepdim=True)) ... 省略复杂实现
        
        # 4. Final Output (Res connection handled in block usually, or here)
        out = self.out_proj(h_cat)
        
        return out + x_main # Block Residual