import scanpy as sc

def process_rna_pipeline(adata, n_top_genes=3000, min_cells=3, target_sum=1e4):
    """
    完全复刻 MultiGATE / Seurat V3 的处理逻辑
    修正：移除了导致不对齐的细胞过滤步骤
    """
    print("   [RNA] Starting preprocessing (Seurat V3 flavor)...")
    print(f"   [RNA] Input shape: {adata.shape}")
    
    # 1. 基础过滤
    # ⚠️ 警告：绝对不要在这里使用 filter_cells，除非你同时去过滤 ATAC 数据
    # sc.pp.filter_cells(adata, min_genes=200)  <-- 必须删掉或注释掉
    
    # 过滤基因是可以的，因为这只改变列数 (Features)，不改变行数 (Cells)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 2. 高变基因筛选 (Seurat V3) - 必须基于 Raw Counts
    print("   [RNA] Selecting highly variable genes (on Raw Counts)...")
    try:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            subset=True # 直接裁剪，保留 Top 3000
        )
    except Exception as e:
        print(f"Error in Seurat V3: {e}")
        raise e
    
    # 3. 标准化 (Normalize)
    print("   [RNA] Normalizing...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # 4. 对数化 (Log1p)
    print("   [RNA] Log transforming...")
    sc.pp.log1p(adata)
    
    # 5. 归一化 (Scale)
    # MultiGate 原文中确实有 Scale 操作
    # print("   [RNA] Scaling...")
    # sc.pp.scale(adata, max_value=10)  # 移除 Z-Score，避免放大噪声
    
    print(f"   [RNA] Processed shape: {adata.shape}")
    return adata