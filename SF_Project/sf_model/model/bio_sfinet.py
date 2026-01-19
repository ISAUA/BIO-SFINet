import torch
import torch.nn as nn
from .encoders import RNA_Encoder, ATAC_Encoder
from .sfib import SFIB

class BioSFINet(nn.Module):
    def __init__(self, config, atac_dim):
        """
        Args:
            atac_dim: 运行时动态获取的 ATAC Peak 数
        """
        super().__init__()
        
        rna_dim = config['model']['rna_in_dim'] # 3000
        hidden_dim = config['model']['hidden_dim'] # 512
        sfib_dim = 128
        
        # 1. Encoders (Phase I)
        self.rna_enc = RNA_Encoder(in_dim=rna_dim, hidden_dim=hidden_dim)
        self.atac_enc = ATAC_Encoder(in_dim=atac_dim, hidden_dim=hidden_dim)
        
        # 2. Projections (Phase II)
        self.rna_proj = nn.Linear(hidden_dim, sfib_dim)
        self.atac_proj = nn.Linear(hidden_dim, sfib_dim)
        self.ln_rna = nn.LayerNorm(sfib_dim)
        self.ln_atac = nn.LayerNorm(sfib_dim)
        
        # 3. Dual Towers (Phase III)
        # Left Tower: ATAC Main
        self.sfib_atac = SFIB(dim=sfib_dim)
        # Right Tower: RNA Main
        self.sfib_rna = SFIB(dim=sfib_dim)
        
        # 4. Decoders (Phase IV)
        self.rna_dec = nn.Linear(sfib_dim, rna_dim)
        self.atac_dec = nn.Linear(sfib_dim, atac_dim)
        
        # 5. Contrastive Head (Phase V)
        self.proj_head = nn.Sequential(
            nn.Linear(sfib_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x_rna, x_atac, edge_index, u_basis):
        # 1. Encode [N, 512]
        h_rna = self.rna_enc(x_rna, edge_index)
        h_atac = self.atac_enc(x_atac)
        
        # 2. Project [N, 128]
        f_rna = self.ln_rna(self.rna_proj(h_rna))
        f_atac = self.ln_atac(self.atac_proj(h_atac))
        
        # 3. Dual SFIB
        # Tower 1 (Left): Main=ATAC, Guide=RNA
        z_atac = self.sfib_atac(x_main=f_atac, x_guide=f_rna, edge_index=edge_index, u_basis=u_basis)
        
        # Tower 2 (Right): Main=RNA, Guide=ATAC
        z_rna = self.sfib_rna(x_main=f_rna, x_guide=f_atac, edge_index=edge_index, u_basis=u_basis)
        
        # 4. Decode
        rec_rna = self.rna_dec(z_rna)
        rec_atac = self.atac_dec(z_atac)
        
        # 5. CLIP Projection
        p_rna = self.proj_head(z_rna)
        p_atac = self.proj_head(z_atac)
        
        return z_rna, z_atac, p_rna, p_atac, rec_rna, rec_atac