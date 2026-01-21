import gzip
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm import tqdm
import re

def parse_gtf_tss(gtf_path):
    """
    解析 GTF 文件 (支持 .gz)，提取基因 TSS
    """
    print(f"Parsing GTF: {gtf_path} ...")
    gene_tss = {}
    
    # 自动判断压缩
    if gtf_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    try:
        with open_func(gtf_path, mode) as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene': continue
                
                try:
                    attr = parts[8]
                    # 适配 gencode 格式
                    if 'gene_name "' in attr:
                        name = attr.split('gene_name "')[1].split('"')[0]
                    elif 'gene_name=' in attr:
                        name = attr.split('gene_name=')[1].split(';')[0]
                    else:
                        continue
                        
                    strand = parts[6]
                    start, end = int(parts[3]), int(parts[4])
                    tss = start if strand == '+' else end
                    chrom = parts[0]
                    if not chrom.startswith('chr'): chrom = 'chr' + chrom
                    gene_tss[name] = (chrom, tss)
                except:
                    continue
    except FileNotFoundError:
        print(f"Error: GTF file not found at {gtf_path}")
        return {}
        
    return gene_tss

def parse_peak_coords(peak_names):
    """
    智能解析 Peak 坐标，支持多种分隔符
    Formats supported:
      - chr1:100-200
      - chr1-100-200
      - chr1,100,200
      - chr1_100_200
    """
    parsed = []
    # 正则表达式提取: (chr...) (数字) (数字)
    # 忽略中间的分隔符
    pattern = re.compile(r"(chr[\w\d]+)[^\w\d]+(\d+)[^\w\d]+(\d+)")
    
    print(f"DEBUG: First 5 peak names in data: {peak_names[:5]}")
    
    for p in peak_names:
        # 尝试正则匹配
        match = pattern.search(p)
        if match:
            chrom = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            center = (start + end) // 2
            parsed.append((chrom, center))
        else:
            # 备用方案：如果名字里没有 chr，可能只是 1:100-200
            try:
                # 尝试通过分割符暴力拆解
                parts = re.split(r'[:\-_,]', p)
                if len(parts) >= 3:
                    c = parts[0] if parts[0].startswith('chr') else 'chr' + parts[0]
                    s = int(parts[1])
                    e = int(parts[2])
                    parsed.append((c, (s+e)//2))
                else:
                    parsed.append((None, None))
            except:
                parsed.append((None, None))
                
    return parsed

def filter_peaks_by_tss(adata, gtf_path, rna_genes, window=100000, n_final=30000):
    """
    根据 RNA 基因的 TSS 筛选物理相关的 Peaks，同时构建 Gene-Peak 掩码矩阵。

    返回:
      filtered_adata: 筛选后的 ATAC AnnData
      gene_peak_mask: scipy.sparse.coo_matrix, shape = (len(rna_genes), n_peaks_kept)
    """
    print(f"Executing Physical Association Filter (Window = +/- {window}bp)...")
    
    # 1. 准备 GTF 数据
    gene_tss = parse_gtf_tss(gtf_path)
    target_genes = set(rna_genes) & set(gene_tss.keys())
    print(f"Matching {len(target_genes)} RNA genes to Peaks...")
    
    if len(target_genes) == 0:
        print("Warning: No matching genes found between RNA data and GTF file!")
        return adata, None

    # 2. 解析 Peak 坐标
    peak_coords = parse_peak_coords(adata.var_names.tolist())
    
    # 构建 DataFrame
    peaks_df = pd.DataFrame(peak_coords, columns=['chrom', 'center'], index=adata.var_names)
    
    # 移除解析失败的 Peaks (None)
    valid_peaks_mask = peaks_df['chrom'].notna()
    peaks_df = peaks_df[valid_peaks_mask]
    
    if len(peaks_df) == 0:
        print("Error: Failed to parse any peak coordinates! Check 'DEBUG' output above.")
        return adata[:, []], None  # Return empty to trigger error safely

    # 3. 筛选逻辑，记录 gene-peak 对应关系
    keep_names_set = set()
    row_indices = []
    col_names = []
    gene_to_row = {g: i for i, g in enumerate(rna_genes)}

    for gene in tqdm(target_genes, desc="Filtering Peaks"):
        chrom, tss = gene_tss[gene]
        
        # 筛选同染色体
        candidates = peaks_df[peaks_df['chrom'] == chrom]
        if candidates.empty:
            continue
        
        # 筛选距离
        matched = candidates[
            (candidates['center'] >= tss - window) & 
            (candidates['center'] <= tss + window)
        ]
        if matched.empty:
            continue

        keep_names_set.update(matched.index)
        row_idx = gene_to_row.get(gene, None)
        if row_idx is None:
            continue
        # 记录每个匹配峰的 gene-row 与 peak-name（列稍后映射）
        for peak_name in matched.index:
            row_indices.append(row_idx)
            col_names.append(peak_name)
        
    print(f"Retained {len(keep_names_set)} peaks out of {adata.shape[1]}")
    
    if len(keep_names_set) == 0:
        print("⚠️ Warning: Physical filter removed ALL peaks. Falling back to retaining TOP variance peaks to prevent crash.")
        fallback_n = min(n_final, adata.shape[1])
        return adata[:, :fallback_n].copy(), None

    # 按原始顺序保留峰，保证列顺序可追踪
    kept_peaks_in_order = [p for p in adata.var_names if p in keep_names_set]
    filtered_adata = adata[:, kept_peaks_in_order].copy()

    # 将 col_names 转为最终列索引
    peak_to_col = {p: i for i, p in enumerate(kept_peaks_in_order)}
    col_indices = [peak_to_col[p] for p in col_names if p in peak_to_col]
    row_indices = [r for p, r in zip(col_names, row_indices) if p in peak_to_col]

    if len(col_indices) == 0:
        # 极端情况：有保留峰但未记录映射（不应发生），安全返回 None
        print("⚠️ Warning: Gene-Peak mapping is empty after filtering.")
        gene_peak_mask = None
    else:
        data = np.ones(len(col_indices), dtype=np.float32)
        shape = (len(rna_genes), len(kept_peaks_in_order))
        gene_peak_mask = sparse.coo_matrix((data, (row_indices, col_indices)), shape=shape, dtype=np.float32)

    return filtered_adata, gene_peak_mask

def custom_tf_idf(adata, eps=1e-6):
    """
    TF-IDF 变换
    """
    print("Applying Custom TF-IDF transform...")
    eps = float(eps)
    if adata.shape[1] == 0:
        print("Error: Input matrix has 0 features (peaks). Skipping TF-IDF.")
        return adata

    X = adata.X
    # 确保数据为 float32
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.array(X, dtype=np.float32)
    adata.X = X

    idf = X.shape[0] / (X.sum(axis=0) + eps) # 加个极小值防止除零
    idf = np.array(idf).flatten()
    
    if sparse.issparse(X):
        sum_per_cell = np.array(X.sum(axis=1)).flatten()
        sum_per_cell[sum_per_cell == 0] = 1
        tf = X.multiply(1.0 / sum_per_cell[:, None])
        adata.X = tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        adata.X = tf * idf
        
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    return adata

def process_atac_pipeline(
    adata,
    rna_genes,
    gtf_path,
    n_global=50000,
    n_final=30000,
    window=100000,
    min_cells=50,
    target_sum=1e4,
    tfidf_eps=1e-6
):
    print("--- Processing ATAC Data ---")
    gene_peak_mask = None
    
    # 1. 基础过滤
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 2. 全局变异度筛选 (flavor=seurat)
    print(f"Selecting top {n_global} peaks by variance...")
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_global)
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"Warning in HVG selection: {e}. Skipping.")

    # 3. TSS 物理筛选
    if gtf_path and rna_genes is not None:
        adata, gene_peak_mask = filter_peaks_by_tss(adata, gtf_path, rna_genes, window=window, n_final=n_final)
        
        # 二次筛选
        if adata.shape[1] > n_final:
            print(f"Downsampling to {n_final} peaks...")
            sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_final)
            hvg_mask = adata.var['highly_variable'].values
            adata = adata[:, hvg_mask].copy()
            if gene_peak_mask is not None:
                # 同步裁剪掩码的列
                selected_idx = np.nonzero(hvg_mask)[0]
                gene_peak_mask = gene_peak_mask.tocsr()[:, selected_idx]

    # 4. TF-IDF
    adata = custom_tf_idf(adata, eps=tfidf_eps)
    
    # 5. 手工 Normalize + Log1p，避免 normalize_total dtype 兼容问题
    if adata.shape[1] > 0:
        if sparse.issparse(adata.X):
            adata.X.data[np.isinf(adata.X.data)] = 0.0
            adata.X.data[np.isnan(adata.X.data)] = 0.0
            X = adata.X.toarray().astype(np.float32)
        else:
            X = np.array(adata.X, dtype=np.float32)
            X[np.isinf(X)] = 0.0
            X[np.isnan(X)] = 0.0

        sums = X.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        X = (X / sums) * float(target_sum)
        X = np.log1p(X)
        adata.X = X
    
    print(f"Final ATAC shape: {adata.shape}")
    return adata, gene_peak_mask