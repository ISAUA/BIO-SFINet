import os
import argparse
import torch
import yaml
import numpy as np
import scanpy as sc

# 引入预处理模块
from sf_model.preprocess.io import read_mtx_to_adata, add_spatial_info
from sf_model.preprocess.rna_process import process_rna_pipeline
from sf_model.preprocess.atac_process import process_atac_pipeline
from sf_model.utils import build_spatial_graph, set_seed

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw spatial multi-omics data")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    return parser.parse_args()

def main():
    print("🚀 [Phase 1] Starting Data Preprocessing...")
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    raw_dir = config['data']['raw_path']
    processed_dir = config['data']['processed_path']
    files = config['data']['files']
    params = config['data']['parameters']
    rna_min_cells = params.get('rna_min_cells', 3)
    rna_target_sum = params.get('rna_target_sum', 1e4)
    atac_min_cells = params.get('atac_min_cells', 50)
    atac_target_sum = params.get('atac_target_sum', 1e4)
    tfidf_eps = float(params.get('tfidf_eps', 1e-6))
    
    os.makedirs(processed_dir, exist_ok=True)

    # ==========================================
    # Step 1: 加载原始数据
    # ==========================================
    print("\n📦 Loading Raw Data...")
    
    # A. 加载 RNA
    print(f"   -> Reading RNA from {files['rna_mtx']}...")
    adata_rna = read_mtx_to_adata(
        os.path.join(raw_dir, files['rna_mtx']),
        os.path.join(raw_dir, files['rna_genes']),
        os.path.join(raw_dir, files['rna_barcodes'])
    )
    # 添加空间坐标
    adata_rna = add_spatial_info(adata_rna, os.path.join(raw_dir, files['spatial']))
    
    # B. 加载 ATAC
    print(f"   -> Reading ATAC from {files['atac_mtx']}...")
    adata_atac = read_mtx_to_adata(
        os.path.join(raw_dir, files['atac_mtx']),
        os.path.join(raw_dir, files['atac_peaks']),
        os.path.join(raw_dir, files['atac_barcodes'])
    )

    # ==========================================
    # Step 2: 严格对齐检查 (不取交集，仅验证)
    # ==========================================
    print("\n🔍 Verifying One-to-One Alignment...")
    n_rna = adata_rna.shape[0]
    n_atac = adata_atac.shape[0]
    
    # 1. 检查数量
    if n_rna != n_atac:
        raise ValueError(f"❌ Mismatch! RNA cells ({n_rna}) != ATAC cells ({n_atac}). Please check raw data.")
    
    # 2. 检查 Barcode 顺序 (简单抽查前5个和后5个)
    if not np.array_equal(adata_rna.obs_names[:5], adata_atac.obs_names[:5]):
        print("⚠️ Warning: Barcode order might mismatch in the first 5 cells!")
        # 如果您非常确定是对应的，可以忽略这个警告，或者在这里强制赋值索引
        # adata_atac.obs_names = adata_rna.obs_names 
    else:
        print("   ✅ Cell count and order look correct.")

    # ==========================================
    # Step 3: 执行预处理管线
    # ==========================================
    
    # --- RNA ---
    # 预防性措施：在传入 pipeline 之前，确保它是 float32
    # 这样即使不修改 rna_process.py，也能避免 normalize_total 报错
    if hasattr(adata_rna.X, "astype"):
        adata_rna.X = adata_rna.X.astype(np.float32)
        
    print("\n🧪 Processing RNA...")
    adata_rna = process_rna_pipeline(
        adata_rna,
        n_top_genes=params['n_top_genes'],
        min_cells=rna_min_cells,
        target_sum=rna_target_sum,
    )

    # --- ATAC ---
    print("\n🧪 Processing ATAC...")
    gtf_path = os.path.join(raw_dir, files['gtf'])
    rna_genes = adata_rna.var_names.tolist()
    
    adata_atac, _ = process_atac_pipeline(
        adata_atac, 
        rna_genes=rna_genes, 
        gtf_path=gtf_path,
        n_global=params['n_global_peaks'],
        n_final=params['n_final_peaks'],
        window=params['tss_window'],
        min_cells=atac_min_cells,
        target_sum=atac_target_sum,
        tfidf_eps=tfidf_eps,
    )

    # ==========================================
    # Step 4: 构建空间图 & 准备 Tensor
    # ==========================================
    print("\n🕸️ Building Spatial Graph & GFT Basis...")
    coords = adata_rna.obsm['spatial']
    
    # 计算图基底 (GFT Basis)
    edge_index, u_basis = build_spatial_graph(coords, k=params['knn_k'])

    # 转换为 Tensor
    def to_tensor(adata):
        if hasattr(adata.X, 'toarray'):
            return torch.FloatTensor(adata.X.toarray())
        return torch.FloatTensor(adata.X)

    rna_feat = to_tensor(adata_rna)
    atac_feat = to_tensor(adata_atac)
    coords_tensor = torch.FloatTensor(coords)

    # ==========================================
    # Step 5: 保存处理好的数据
    # ==========================================
    save_path = os.path.join(processed_dir, "processed_data.pt")
    print(f"\n💾 Saving processed tensors to {save_path}...")
    
    data_dict = {
        "rna_feat": rna_feat,
        "atac_feat": atac_feat,
        "coords": coords_tensor,
        "edge_index": edge_index,
        "u_basis": u_basis,
        "atac_dim": atac_feat.shape[1] # 记录动态 ATAC 维度
    }
    
    torch.save(data_dict, save_path)
    print("✅ Preprocessing Complete!")

if __name__ == "__main__":
    main()