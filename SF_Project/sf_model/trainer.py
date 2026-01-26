import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from .utils import CLIPLoss

# 1. 新增: 加权 MSE Loss 类
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=10.0): # 推荐权重 10-20
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        # 生成权重: 如果 target > 0 (有真实信号)，给予高权重
        weights = torch.ones_like(target)
        weights[target > 0] = self.pos_weight 
        return (loss * weights).mean()

class SFTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        clip_temp = float(config['train'].get('clip_temperature', 0.1))
        self.clip_criterion = CLIPLoss(temperature=clip_temp).to(device)
        
        # 2. 修改: 初始化 WeightedMSELoss
        # 推荐: RNA 权重 5-10, ATAC 权重 10-20 (因为 ATAC 更稀疏)
        self.criterion_rna = WeightedMSELoss(pos_weight=10.0).to(device)
        self.criterion_atac = WeightedMSELoss(pos_weight=20.0).to(device)

        params = list(self.model.parameters()) + list(self.clip_criterion.parameters())

        self.optimizer = optim.AdamW(
            params,
            lr=float(config['train']['learning_rate']),
            weight_decay=float(config['train']['weight_decay'])
        )
        self.save_dir = config['project']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger("SFTrainer")
        self.save_every = config['train'].get('save_every', 50)
        self.log_interval = int(config['train'].get('log_interval', 10))
        self.best_name = config['train'].get('best_name', 'ckpt_best.pth')
        self.best_path = os.path.join(self.save_dir, self.best_name)

    def train_epoch(self, rna_feat, atac_feat, edge_index, u_basis):
        self.model.train()
        self.optimizer.zero_grad()
        
        rna_feat = rna_feat.to(self.device)
        atac_feat = atac_feat.to(self.device)
        edge_index = edge_index.to(self.device)
        u_basis = u_basis.to(self.device)
        
        z_rna, z_atac, p_rna, p_atac, rec_rna, rec_atac = self.model(
            rna_feat, atac_feat, edge_index, u_basis
        )
        
        # 3. 修改: 使用 Weighted Loss 计算
        loss_rec_rna = self.criterion_rna(rec_rna, rna_feat)
        loss_rec_atac = self.criterion_atac(rec_atac, atac_feat)
        
        loss_clip = self.clip_criterion(p_rna, p_atac)
        
        lambda_r = self.config['train'].get('lambda_rna', 1.0)
        lambda_a = self.config['train'].get('lambda_atac', 1.0)
        lambda_c = self.config['train'].get('lambda_clip', 0.1)
        
        total_loss = lambda_r * loss_rec_rna + lambda_a * loss_rec_atac + lambda_c * loss_clip
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total": total_loss.item(),
            "rec_rna": loss_rec_rna.item(),
            "rec_atac": loss_rec_atac.item(),
            "clip": loss_clip.item()
        }
    
    # run 函数保持不变...
    def run(self, rna_data, atac_data, edge_index, u_basis):
        # ... (保持原样)
        epochs = self.config['train']['epochs']
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(rna_data, atac_data, edge_index, u_basis)

            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), self.best_path)

            if epoch % self.log_interval == 0:
                best_display = f"{best_loss:.4f}" if best_loss < float('inf') else "N/A"
                print(
                    f"Epoch {epoch:03d} | total {metrics['total']:.4f} | "
                    f"rec_rna {metrics['rec_rna']:.4f} | rec_atac {metrics['rec_atac']:.4f} | "
                    f"clip {metrics['clip']:.4f} | best {best_display}"
                )

            if epoch % self.save_every == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"ckpt_{epoch}.pth"))