import argparse
import os
import subprocess
import sys

# 数据集到配置文件的映射，新增数据集时在此注册
DATASET_CONFIG = {
    "human": "configs/config_human.yaml",
    "mouse": "configs/config_mouse.yaml",
}

ALL_STEPS = ["preprocess", "train", "evaluate"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run preprocessing, training, and evaluation for Bio-SFINet")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG.keys()), default="human", help="选择数据集")
    parser.add_argument(
        "--steps",
        default="preprocess,train,evaluate",
        help="要执行的步骤，逗号分隔，可选 preprocess/train/evaluate，例如: preprocess,train",
    )
    parser.add_argument("--checkpoint", default=None, help="评估阶段使用的 checkpoint key 或文件名，可选")
    parser.add_argument("--resolution", type=float, default=None, help="评估阶段的 Leiden 分辨率，可选")
    return parser.parse_args()


def run_cmd(label, cmd):
    print(f"\n===== {label}: {' '.join(cmd)} =====")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    args = parse_args()

    if args.dataset not in DATASET_CONFIG:
        print(f"未注册的数据集: {args.dataset}. 请在 DATASET_CONFIG 中添加映射。")
        sys.exit(1)

    config_path = DATASET_CONFIG[args.dataset]
    base_dir = os.path.abspath(os.path.dirname(__file__))

    requested_steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    for step in requested_steps:
        if step not in ALL_STEPS:
            print(f"不支持的步骤: {step}. 仅支持 {ALL_STEPS}")
            sys.exit(1)

    steps = []

    if "preprocess" in requested_steps:
        steps.append((
            "Preprocess",
            [sys.executable, os.path.join(base_dir, "run_preprocess.py"), "--config", config_path],
        ))

    if "train" in requested_steps:
        steps.append((
            "Train",
            [sys.executable, os.path.join(base_dir, "run_train.py"), "--config", config_path],
        ))

    if "evaluate" in requested_steps:
        eval_cmd = [sys.executable, os.path.join(base_dir, "run_evaluate.py"), "--config", config_path]
        if args.checkpoint:
            eval_cmd.extend(["--checkpoint", args.checkpoint])
        if args.resolution is not None:
            eval_cmd.extend(["--resolution", str(args.resolution)])
        steps.append(("Evaluate", eval_cmd))

    if not steps:
        print("未选择任何步骤，已退出。")
        return

    print(f"使用配置: {config_path}")
    print(f"执行顺序: {[name for name, _ in steps]}")

    for name, cmd in steps:
        run_cmd(name, cmd)


if __name__ == "__main__":
    main()
