import torch
from torch.utils.data import DataLoader
from model import Uniformer_non_contrast
from h5dataset import H5Dataset
from UniFormermain.video_classification.slowfast.config.defaults import assert_and_infer_cfg
from UniFormermain.video_classification.slowfast.utils.parser import load_config, parse_args
import os
import warnings
import csv
import numpy as np
from types import SimpleNamespace

@torch.no_grad()
def predict_from_h5_dir(
    h5_dir: str,
    csv_output_path: str,
    checkpoint_dir: str,
    cfg_file: str = r"E:/Wangsiqi2/CARE-liver/config.yaml",
    setting: str = "Noncontrast",
    num_workers: int = 0,   
):
    warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader")

    args = SimpleNamespace(cfg_file=cfg_file, opts=None)
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Uniformer_non_contrast(cfg).to(device)
    model.eval()

    # 收集 checkpoint
    checkpoint_paths = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    )
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No .pth checkpoints found in: {checkpoint_dir}")

    # 数据
    test_dataset = H5Dataset(h5_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    all_probs = {}   # case_id -> list of (2,) array, one per fold
    all_case_ids = []

    for fold, ckpt_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"✅ 加载模型权重: {ckpt_path}")

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        for batch in test_loader:
            inputs = [batch[k].to(device) for k in ["t1", "t2", "dwi"]]
            outputs = model(inputs)  # (1,2)
            probs = torch.sigmoid(outputs).cpu().numpy()  # (1,2)

            filenames = batch["filename"]
            case_id = os.path.basename(filenames[0]).replace(".h5", "")

            if case_id not in all_probs:
                all_probs[case_id] = []
            all_probs[case_id].append(probs[0])

            if fold == 0:
                all_case_ids.append(case_id)

    # 写 CSV
    os.makedirs(os.path.dirname(csv_output_path) or ".", exist_ok=True)
    fold_log_path = os.path.join(os.path.dirname(csv_output_path) or ".", "fold.txt")

    with open(csv_output_path, mode="w", newline="", encoding="utf-8") as f_csv, \
         open(fold_log_path, mode="w", encoding="utf-8") as f_log:

        writer = csv.writer(f_csv)
        writer.writerow(["Case", "Setting", "Subtask1_prob_S4", "Subtask2_prob_S1"])

        for case_id in sorted(all_case_ids):
            prob_array = np.array(all_probs[case_id])  # (num_folds, 2)

            probs_s4 = prob_array[:, 0]  # Subtask1: S4 vs others
            probs_s1 = prob_array[:, 1]  # Subtask2: S1 vs others

            final_s4 = float(probs_s4.mean())
            final_s1 = float(probs_s1.mean())

            writer.writerow([case_id, setting, final_s4, final_s1])

            # 记录日志
            f_log.write(f"[{case_id}]\n")
            f_log.write(f"S4 fold probs: {probs_s4.tolist()}\n")
            f_log.write(f"S1 fold probs: {probs_s1.tolist()}\n")
            f_log.write(f"Final S4: {final_s4:.6f}, Final S1: {final_s1:.6f}\n")
            f_log.write("--------\n")

    print(f"[✓] 最终预测写入: {csv_output_path}")
    print(f"[📄] Fold日志写入: {fold_log_path}")

@torch.no_grad()
def predict_from_h5_dir_val(h5_dir: str, csv_output_path: str, checkpoint_dir: str):
    warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader")

    # 配置
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # 设备与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Uniformer_non_contrast(cfg).to(device)

    # checkpoint 列表（5折）
    checkpoint_paths = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth")
    ]
    checkpoint_paths.sort()
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No .pth checkpoints found in: {checkpoint_dir}")

    # 测试集
    test_dataset = H5Dataset(h5_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # case_id -> list of (2,) probs from each fold
    all_probs = {}
    all_case_ids = []

    # 逐折推理
    for fold, ckpt_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"加载模型权重: {ckpt_path}")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        for batch in test_loader:
            # 只输入 [t1, t2, dwi]
            inputs = [batch[k].to(device) for k in ["t1", "t2", "dwi"]]

            outputs = model(inputs)                # (B, 2)
            probs = torch.sigmoid(outputs).cpu().numpy()  # (B, 2)

            filenames = batch["filename"]
            for i in range(len(filenames)):
                case_id = os.path.basename(filenames[i]).replace(".h5", "")
                if case_id not in all_probs:
                    all_probs[case_id] = []
                all_probs[case_id].append(probs[i])

                if fold == 0:
                    all_case_ids.append(case_id)

    # 写 CSV（只保留预测结果）
    out_dir = os.path.dirname(csv_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_output_path, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Case", "Setting", "Subtask1_prob_S4", "Subtask2_prob_S1"])

        for case_id in sorted(all_case_ids):
            prob_array = np.array(all_probs[case_id])  # shape: (n_folds, 2)
            probs_s4 = prob_array[:, 0]
            probs_s1 = prob_array[:, 1]

            # 5折平均（最简单稳定）
            final_s4 = float(probs_s4.mean())
            final_s1 = float(probs_s1.mean())

            writer.writerow([case_id, "NonContrast", final_s4, final_s1])

    print(f"[✓] 最终预测写入: {csv_output_path}")


if __name__ == "__main__":
    h5_dir = r"E:\BaiduNetdiskDownload\LiQA_val\h5_output_dataset"
    csv_output_path = r".\test_predictions_noncontrast.csv"
    checkpoint_dir = r"E:\Wangsiqi2\CARE-liver\output\Liverclass\260112_154508\fold_epoch"
    predict_from_h5_dir(h5_dir, csv_output_path, checkpoint_dir)
