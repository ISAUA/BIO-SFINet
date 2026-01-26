import torch
import torch.nn as nn
from .encoders import RNA_Encoder, ATAC_Encoder
from .sfib import SFIB

# ==========================================
# 1. 新增组件: 从 GraphTransformer 迁移的深度解码器
# ==========================================

class ResidualBlock(nn.Module):
    """标准残差块（Pre-LN）：两层子层并做残差连接。"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim)

        def _sublayer() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.block = nn.Sequential(
            _sublayer(),
            _sublayer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(x)
        return x


class DeepDecoder(nn.Module):
    """Residual Deep Decoder (Projection + Residual Stack + Output)."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        n_blocks = int(n_blocks)

        # 1. 投影到高维隐空间
        self.proj = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. 残差堆叠
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        
        # 3. 输出映射
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return self.out(x)


# ==========================================
# 2. 修改后的 BioSFINet 主模型 (双塔版本)
# ==========================================

class BioSFINet(nn.Module):
    def __init__(self, config, atac_dim):
        """
        Args:
            atac_dim: 运行时动态获取的 ATAC Peak 数
        """
        super().__init__()
        
        model_cfg = config['model']
        rna_dim = model_cfg['rna_in_dim']
        hidden_dim = model_cfg['hidden_dim']
        sfib_dim = model_cfg.get('sfib_dim', 128)
        rna_heads = model_cfg.get('rna_n_heads', model_cfg.get('n_heads', 4))
        rna_dropout = model_cfg.get('rna_dropout', model_cfg.get('dropout', 0.1))
        atac_dropout = model_cfg.get('atac_dropout', model_cfg.get('dropout', 0.1))
        proj_hidden = model_cfg.get('proj_hidden_dim', 64)
        proj_output = model_cfg.get('proj_output_dim', 64)
        
        # 1. Encoders (Phase I) - 保持不变
        self.rna_enc = RNA_Encoder(in_dim=rna_dim, hidden_dim=hidden_dim, n_heads=rna_heads, dropout=rna_dropout)
        self.atac_enc = ATAC_Encoder(in_dim=atac_dim, hidden_dim=hidden_dim, dropout=atac_dropout)
        
        # 2. Projections (Phase II) - 保持不变
        self.rna_proj = nn.Linear(hidden_dim, sfib_dim)
        self.atac_proj = nn.Linear(hidden_dim, sfib_dim)
        self.ln_rna = nn.LayerNorm(sfib_dim)
        self.ln_atac = nn.LayerNorm(sfib_dim)
        
        # 3. Dual Towers (Phase III) - 保持不变
        # Left Tower: ATAC Main
        self.sfib_atac = SFIB(dim=sfib_dim)
        # Right Tower: RNA Main
        self.sfib_rna = SFIB(dim=sfib_dim)
        
        # 4. Decoders (Phase IV) - [核心修改]
        # 使用 DeepDecoder 替换简单的 Linear
        
        # RNA Decoder: 1024 hidden dim, 3 layers
        self.rna_dec = DeepDecoder(
            in_dim=sfib_dim,
            out_dim=rna_dim,
            hidden_dim=512,
            n_blocks=1,
            dropout=rna_dropout
        )
        
        # ATAC Decoder: 2048 hidden dim, 3 layers (ATAC 更稀疏需要更大容量)
        self.atac_dec = DeepDecoder(
            in_dim=sfib_dim,
            out_dim=atac_dim,
            hidden_dim=512,
            n_blocks=1,
            dropout=atac_dropout
        )
        
        # 5. Contrastive Head (Phase V) - 保持不变
        self.proj_head = nn.Sequential(
            nn.Linear(sfib_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_output)
        )

    def forward(self, x_rna, x_atac, edge_index, u_basis):
        # 1. Encode [N, 512]
        h_rna = self.rna_enc(x_rna, edge_index)
        h_atac = self.atac_enc(x_atac)
        
        # 2. Project [N, 128]
        f_rna = self.ln_rna(self.rna_proj(h_rna))
        f_atac = self.ln_atac(self.atac_proj(h_atac))
        
        # 3. Dual SFIB
        # Tower 1 (Left): Main=ATAC, Guide=RNA -> z_atac
        z_atac = self.sfib_atac(x_main=f_atac, x_guide=f_rna, edge_index=edge_index, u_basis=u_basis)
        
        # Tower 2 (Right): Main=RNA, Guide=ATAC -> z_rna
        z_rna = self.sfib_rna(x_main=f_rna, x_guide=f_atac, edge_index=edge_index, u_basis=u_basis)
        
        # 4. Decode
        # 调用接口与 nn.Linear 一致，无需修改 forward 逻辑
        rec_rna = self.rna_dec(z_rna)
        rec_atac = self.atac_dec(z_atac)
        
        # 5. CLIP Projection
        p_rna = self.proj_head(z_rna)
        p_atac = self.proj_head(z_atac)
        
        return z_rna, z_atac, p_rna, p_atac, rec_rna, rec_atac