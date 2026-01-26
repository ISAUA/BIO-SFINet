# BIO-SFIB Pipeline 使用说明

本项目提供统一入口脚本用于预处理、训练与评估多组学空间数据，可在不同数据集之间灵活切换，目前内置 human 与 P22 mouse。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1) 全流程运行

```bash
python run_pipeline.py --dataset human
python run_pipeline.py --dataset mouse
```

### 2) 仅运行部分阶段

`--steps` 逗号分隔，可选 `preprocess`, `train`, `evaluate`。

示例：仅预处理
```bash
python run_pipeline.py --dataset mouse --steps preprocess
```

示例：预处理 + 训练
```bash
python run_pipeline.py --dataset mouse --steps preprocess,train
```

### 3) 评估阶段附加参数

- `--checkpoint`: 指定评估使用的 checkpoint key 或文件名（默认读取配置文件 eval.checkpoint）。
- `--resolution`: 调整 Leiden 聚类分辨率，示例：`--resolution 0.6`。

示例：仅评估并指定 checkpoint
```bash
python run_pipeline.py --dataset mouse --steps evaluate --checkpoint ckpt_best.pth --resolution 0.6
```

## 数据集配置与扩展

- 数据集配置映射在 run_pipeline.py 的 `DATASET_CONFIG` 中注册。
- 内置配置文件：
	- human: configs/config_human.yaml
	- mouse: configs/config_mouse.yaml（P22 小鼠脑数据，结果输出至 results/mouse）。
- 如需新增数据集，准备对应 YAML 配置并加入映射后即可使用 `--dataset <new>` 运行。

## 目录约定

- 原始数据：data/raw/<dataset>/
- 预处理结果：data/processed/<dataset>/processed_data.pt
- 训练输出：results/<dataset>/checkpoints/
- 评估输出：results/<dataset>/predictions/ 与 results/<dataset>/figures/

## 常见命令速查

- 全流程（human）：`python run_pipeline.py --dataset human`
- 全流程（mouse）：`python run_pipeline.py --dataset mouse`
- 仅预处理：`python run_pipeline.py --dataset human --steps preprocess`
- 仅训练：`python run_pipeline.py --dataset human --steps train`
- 仅评估：`python run_pipeline.py --dataset human --steps evaluate --checkpoint ckpt_best.pth`
- 评估和绘图参数：`python run_pipeline.py --dataset human --steps evaluate --checkpoint ckpt_best.pth --resolution 0.9`