import os
import torch
import yaml
import numpy as np
import scanpy as sc

# 引入已有的模型和工具
from sf_model.model.bio_sfinet import BioSFINet
from sf_model.trainer import SFTrainer
from sf_model.utils import build_spatial_graph  # 使用 utils 里的图构建

# 引入你的预处理模块
from sf_model.preprocess.io import read_mtx_to_adata, add_spatial_info
from sf_model.preprocess.rna_process import process_rna_pipeline
from sf_model.preprocess.atac_process import process_atac_pipeline

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_preprocess_data(config):
    print("🚀 [1/4] Loading & Preprocessing Data...")
    raw_dir = config['data']['raw_path']
    files = config['data']['files']
    params = config['data']['parameters']

    # --- 1. Load Data ---
    print("   -> Reading MTX files...")
    adata_rna = read_mtx_to_adata(
        os.path.join(raw_dir, files['rna_mtx']),
        os.path.join(raw_dir, files['rna_genes']),
        os.path.join(raw_dir, files['rna_barcodes'])
    )
    adata_atac = read_mtx_to_adata(
        os.path.join(raw_dir, files['atac_mtx']),
        os.path.join(raw_dir, files['atac_peaks']),
        os.path.join(raw_dir, files['atac_barcodes'])
    )

    # Add Spatial Info
    adata_rna = add_spatial_info(adata_rna, os.path.join(raw_dir, files['spatial']))
    
    # --- 2. Alignment (Intersection) ---
    common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
    print(f"   -> Aligning: {len(common_cells)} common cells.")
    adata_rna = adata_rna[common_cells].copy()
    adata_atac = adata_atac[common_cells].copy()
    
    # --- 3. RNA Processing ---
    # 使用你的 rna_process.py
    adata_rna = process_rna_pipeline(adata_rna, n_top_genes=params['n_top_genes'])

    # --- 4. ATAC Processing ---
    # 使用你的 atac_process.py (包含 TSS 筛选 + TF-IDF)
    # 注意: 需要 GTF 文件路径
    gtf_path = os.path.join(raw_dir, files['gtf'])
    rna_genes = adata_rna.var_names.tolist() # 用于 TSS 筛选的基因列表
    
    adata_atac, _ = process_atac_pipeline(
        adata_atac, 
        rna_genes=rna_genes, 
        gtf_path=gtf_path,
        n_global=params['n_global_peaks'],
        n_final=params['n_final_peaks'],
        window=params['tss_window']
    )

    # --- 5. Prepare Tensors ---
    print("   -> Converting to Tensors...")
    # 确保是 Dense Tensor
    def to_tensor(adata):
        if hasattr(adata.X, 'toarray'):
            return torch.FloatTensor(adata.X.toarray())
        return torch.FloatTensor(adata.X)

    rna_feat = to_tensor(adata_rna)
    atac_feat = to_tensor(adata_atac)
    coords = adata_rna.obsm['spatial'] # numpy
    
    return rna_feat, atac_feat, coords

def main():
    # 1. Config
    config = load_config("configs/config_human.yaml")
    
    # 2. Data
    rna_feat, atac_feat, coords = load_and_preprocess_data(config)
    
    # 3. Graph & GFT Basis
    print("🚀 [3/4] Building Spatial Graph & GFT Basis...")
    # 调用 sf_model/utils.py 中的函数
    edge_index, u_basis = build_spatial_graph(coords, k=config['data']['parameters']['knn_k'])
    print(f"   -> Basis shape: {u_basis.shape}")
    
    # 4. Model
    print("🚀 [4/4] Initializing Bio-SFINet...")
    atac_dim = atac_feat.shape[1]
    print(f"   -> Dynamic ATAC Dimension: {atac_dim}")
    
    model = BioSFINet(config, atac_dim=atac_dim)
    
    # 5. Training
    trainer = SFTrainer(model, config)
    print("\n🟢 STARTING TRAINING...")
    trainer.run(rna_feat, atac_feat, edge_index, u_basis)

if __name__ == "__main__":
    main()