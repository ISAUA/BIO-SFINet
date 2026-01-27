import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from .utils import CLIPLoss

# 1. 加权 MSE Loss 类 (保持不变)
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=10.0): 
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

        self.train_cfg = config['train']
        clip_temp = float(self.train_cfg.get('clip_temperature', 0.1))
        self.clip_criterion = CLIPLoss(temperature=clip_temp).to(device)
        
        # 初始化 Loss (保持不变，WeightedMSE 对于深层网络至关重要)
        pos_weight_rna = float(self.train_cfg.get('pos_weight_rna', 10.0))
        pos_weight_atac = float(self.train_cfg.get('pos_weight_atac', 20.0))
        self.criterion_rna = WeightedMSELoss(pos_weight=pos_weight_rna).to(device)
        self.criterion_atac = WeightedMSELoss(pos_weight=pos_weight_atac).to(device)

        params = list(self.model.parameters()) + list(self.clip_criterion.parameters())

        lr_high = float(self.train_cfg['learning_rate'])
        lr_low = float(self.train_cfg.get('learning_rate_low', lr_high * 0.1))
        lr_switch = int(self.train_cfg.get('lr_switch_epoch', 500))

        self.optimizer = optim.AdamW(
            params,
            lr=lr_high,
            weight_decay=float(config['train']['weight_decay'])
        )

        # 两段式学习率：前 lr_switch 轮用 lr_high，之后降到 lr_low
        def lr_lambda(epoch_idx: int):
            return 1.0 if epoch_idx < lr_switch else lr_low / lr_high

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.lr_switch = lr_switch
        self.lr_high = lr_high
        self.lr_low = lr_low
        self.save_dir = config['project']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger("SFTrainer")
        self.save_every = self.train_cfg.get('save_every', 50)
        self.log_interval = int(self.train_cfg.get('log_interval', 10))
        self.best_name = self.train_cfg.get('best_name', 'ckpt_best.pth')
        self.best_path = os.path.join(self.save_dir, self.best_name)

    def train_epoch(self, rna_feat, atac_feat, edge_index, u_basis):
        self.model.train()
        self.optimizer.zero_grad()
        
        rna_feat = rna_feat.to(self.device)
        atac_feat = atac_feat.to(self.device)
        edge_index = edge_index.to(self.device)
        u_basis = u_basis.to(self.device)
        
        # Forward pass
        z_rna, z_atac, p_rna, p_atac, rec_rna, rec_atac = self.model(
            rna_feat, atac_feat, edge_index, u_basis
        )
        
        # 1. 重构损失
        loss_rec_rna = self.criterion_rna(rec_rna, rna_feat)
        loss_rec_atac = self.criterion_atac(rec_atac, atac_feat)
        
        # 2. 对齐损失
        loss_clip = self.clip_criterion(p_rna, p_atac)
        
        lambda_r = float(self.train_cfg.get('lambda_rna', 1.0))
        lambda_a = float(self.train_cfg.get('lambda_atac', 1.0))
        lambda_c = float(self.train_cfg.get('lambda_clip', 0.1))
        
        total_loss = lambda_r * loss_rec_rna + lambda_a * loss_rec_atac + lambda_c * loss_clip
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total": total_loss.item(),
            "rec_rna": loss_rec_rna.item(),
            "rec_atac": loss_rec_atac.item(),
            "clip": loss_clip.item()
        }

    def run(self, rna_data, atac_data, edge_index, u_basis):
        epochs = self.config['train']['epochs']
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(rna_data, atac_data, edge_index, u_basis)

            # 更新学习率调度
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), self.best_path)

            if epoch % self.log_interval == 0:
                best_display = f"{best_loss:.4f}" if best_loss < float('inf') else "N/A"
                print(
                    f"Epoch {epoch:03d} | total {metrics['total']:.4f} | "
                    f"rec_rna {metrics['rec_rna']:.4f} | rec_atac {metrics['rec_atac']:.4f} | "
                    f"clip {metrics['clip']:.4f} | lr {current_lr:.2e} | best {best_display}"
                )

            if epoch % self.save_every == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"ckpt_{epoch}.pth"))