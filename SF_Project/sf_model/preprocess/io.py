import pandas as pd
import anndata as ad
from scipy import io, sparse
import os

def read_mtx_to_adata(mtx_path, features_path, barcodes_path, transpose=True):
    """
    读取 MTX 文件并构建 AnnData (逻辑源自 prepare_adata.py)
    """
    print(f"Reading data from: {mtx_path}")
    
    # 1. 读取矩阵 (prepare_adata.py logic)
    mat = io.mmread(mtx_path)
    
    # 自动处理转置 (Your script: rna_mat = rna_mat.transpose())
    if transpose:
        mat = mat.transpose()
    
    # 转为 CSR 格式 (Your script: rna_mat = csr_matrix(rna_mat))
    mat = sparse.csr_matrix(mat)
    
    # 2. 读取元数据
    # header=None 对应您脚本中的 pd.read_csv(..., header=None)
    features = pd.read_csv(features_path, header=None, sep='\t')
    barcodes = pd.read_csv(barcodes_path, header=None, sep=',') # 注意您脚本中barcodes用的是逗号分隔
    
    # 3. 构建 AnnData
    adata = ad.AnnData(mat)
    adata.obs_names = barcodes.iloc[:, 0].astype(str)
    adata.var_names = features.iloc[:, 0].astype(str)
    
    # 确保唯一性 (Scanpy 常用操作)
    adata.var_names_make_unique()
    
    return adata

def add_spatial_info(adata, spatial_path):
    """
    添加空间坐标 (逻辑源自 prepare_adata.py)
    """
    # Your script: spatial = pd.read_csv("position.tsv", sep=',', index_col=0)
    spatial_df = pd.read_csv(spatial_path, sep=',', index_col=0)
    
    # 取交集 (防止报错)
    common_cells = adata.obs_names.intersection(spatial_df.index)
    if len(common_cells) < len(adata):
        print(f"Warning: Only {len(common_cells)} cells have spatial coords.")
        adata = adata[common_cells, :].copy()
    
    # Your script: coor_df = spatial.loc[..., ['imagecol','imagerow']]
    coords = spatial_df.loc[adata.obs_names, ['imagecol', 'imagerow']].values
    adata.obsm['spatial'] = coords
    
    return adata