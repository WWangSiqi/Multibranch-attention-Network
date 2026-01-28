import torch
import torch.nn as nn
import torch.nn.functional as F
from module.bottle import TransformerBlock
import os, sys
slowfast_path = os.path.abspath("UniFormermain/video_classification/")
sys.path.insert(0, slowfast_path)
from UniFormermain.video_classification.slowfast.models.uniformer_light_fp32 import Uniformer_light_fp32
from UniFormermain.video_classification.slowfast.config.defaults import assert_and_infer_cfg
from UniFormermain.video_classification.slowfast.utils.parser import load_config, parse_args
from copy import deepcopy


class Uniformer_non_contrast(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super().__init__()
        self.cfg = cfg

        # 只保留一个 encoder（输入为 [t1, t2, dwi] 拼成 3 通道）
        self.encoder = Uniformer_light_fp32(deepcopy(cfg))
        self.load_pretrained_weights(self.encoder)


        # 分类头：原来是 5*512 -> 1024
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def load_pretrained_weights(self, model):
        """加载通用预训练权重，并打印加载情况"""
        checkpoint_path = r"E:\Wangsiqi2\CARE-liver\path_to_models\uniformer_xs32_192_k400.pth"
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model_state = model.state_dict()
        new_state_dict = {}

        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"⚠️ 跳过参数: {k}")

        model.load_state_dict(new_state_dict, strict=False)

        # 是否训练 encoder：你原来是 True（不冻结）
        for p in model.parameters():
            p.requires_grad = True

    def forward(self, inputs):
        """
        inputs 现在只接受 (t1, t2, dwi)
        t1/t2/dwi: (B, 1, H, W) 之类的张量
        """
        t1, t2, dwi = inputs  # ✅ 只保留 3 个输入

        x = torch.cat([t1, t2, dwi], dim=1)  # (B, 3, H, W)

        # 提特征
        _, feat = self.encoder.forward_features(x)
        # 常见输出 feat: (B, C, H', W') 或 (B, C, T) 等

        # 统一做 GAP：先 flatten 空间维，再对 token/空间维求均值
        if feat.dim() >= 3:
            feat = feat.flatten(2)           # (B, C, T)
            feat_gap = feat.mean(dim=2)      # (B, C)
        else:
            # 若 feat 已经是 (B, C)
            feat_gap = feat

        # MLP head
        out = F.relu(self.fc1(feat_gap))
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu(self.fc2(out))

        out = self.fc3(out)
        return out

if __name__ == "__main__":
    inputs = [
        torch.randn(4, 1, 32, 224, 224),  # T1
        torch.randn(4, 1, 32, 224, 224),  # T2
        torch.randn(4, 1, 32, 224, 224),  # DWI
    ]
    # outputs = model(inputs)
    # print(outputs.shape)  # 应该是 [4, 4]
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    model = Uniformer_non_contrast(cfg)
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征的形状