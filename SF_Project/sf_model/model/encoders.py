import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RNA_Encoder(nn.Module):
    """
    RNA 模态编码器 (Guidance / Structure Stream)
    
    设计理念:
    利用 GAT (Graph Attention Network) 聚合空间邻域信息，
    将基因表达特征转化为蕴含局部空间结构的隐变量。
    这对应于 SFITNET 中的 "相位 (Phase/Structure)" 提取分支。
    """
    def __init__(self, in_dim=3000, hidden_dim=512, n_heads=4, dropout=0.1):
        super(RNA_Encoder, self).__init__()
        
        # GATConv: 自动处理输入维度到隐藏维度的映射
        # 我们使用多头注意力机制来增强对空间模式的捕获能力
        # 输出维度将是 hidden_dim * n_heads，所以我们需要再做一次投影或平均
        
        # 第一层 GAT: 3000 -> hidden_dim
        # concat=False 表示我们将多头的输出取平均，而不是拼接，保持维度为 hidden_dim
        self.gat1 = GATConv(in_dim, hidden_dim, heads=n_heads, concat=False, dropout=dropout)
        
        # 可选：第二层 GAT (加深网络以捕获更广的感受野)
        # self.gat2 = GATConv(hidden_dim, hidden_dim, heads=n_heads, concat=False, dropout=dropout)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU() # GAT 常用激活函数

    def forward(self, x, edge_index):
        """
        Args:
            x: RNA 特征矩阵 [Num_Cells, 3000]
            edge_index: 空间邻接图边索引 [2, Num_Edges]
        Returns:
            h_rna: 编码后的 RNA 特征 [Num_Cells, 512]
        """
        # 1. 图注意力卷积
        x = self.gat1(x, edge_index)
        
        # 2. 激活与归一化
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class ATAC_Encoder(nn.Module):
    """
    ATAC 模态编码器 (Target / Content Stream)
    
    设计理念:
    利用 MLP (Multilayer Perceptron) 提取细胞特异性的表观特征。
    不使用 GCN/GAT，以防止稀疏的特异性信号被邻域过度平滑 (Over-smoothing)。
    这对应于 SFITNET 中的 "幅度 (Amplitude/Content)" 提取分支。
    """
    def __init__(self, in_dim, hidden_dim=512, dropout=0.1):
        super(ATAC_Encoder, self).__init__()
        
        # 这里的 in_dim 是动态的，通常是 ~8000-10000 (取决于预处理筛选结果)
        
        self.mlp = nn.Sequential(
            # 第一层投影: High-Dim -> Hidden-Dim
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(), # GELU 在 Transformer 类模型中表现通常优于 ReLU
            nn.Dropout(dropout),
            
            # 第二层特征提取 (可选，增强非线性能力)
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: ATAC 特征矩阵 [Num_Cells, ~9000]
        Returns:
            h_atac: 编码后的 ATAC 特征 [Num_Cells, 512]
        """
        return self.mlp(x)