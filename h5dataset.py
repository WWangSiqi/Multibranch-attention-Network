import os
import random
import h5py
import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, h5_folder, split='train', fold_idx=0, num_folds=5, seed=2025):
        self.h5_files = sorted([
            os.path.join(h5_folder, f)
            for f in os.listdir(h5_folder)
            if f.endswith('.h5')
        ])
        
        # 统一打乱顺序
        random.seed(seed)
        random.shuffle(self.h5_files)

        # 计算每折的大小
        fold_size = len(self.h5_files) // num_folds
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx != num_folds - 1 else len(self.h5_files)

        if split == 'train':
            self.h5_files = self.h5_files[:val_start] + self.h5_files[val_end:]
        elif split == 'val':
            self.h5_files = self.h5_files[val_start:val_end]
        elif split == 'test':
            pass  # 全部数据
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.transform = self.get_transform(split)

    def get_transform(self, split):
        transform_3d = {
            "train": tio.Compose([
                tio.Resize((32, 192, 192)),
                tio.RandomFlip(axes=('LR',), flip_probability=0.5),     # 随机左右翻转
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10,          # 随机旋转和平移
                                 translation=2, p=0.75),
                tio.RandomNoise(mean=0, std=0.03, p=0.5),                 # 高斯噪声
                # tio.RandomBlur(std=(0.1, 1.0), p=0.3),                   # 高斯模糊
            ]),
            "val": tio.Resize((32, 192, 192)),
            "test": tio.Resize((32, 192, 192)),
        }
        return transform_3d.get(split, None)

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        file_path = self.h5_files[index]
        filename = os.path.basename(file_path)

        with h5py.File(file_path, 'r') as h5file:
            t1 = h5file['seq_1'][:]
            t2 = h5file['seq_2'][:]
            dwi = h5file['seq_3'][:]
            ged1 = h5file['seq_4'][:]
            ged2 = h5file['seq_5'][:]
            ged3 = h5file['seq_6'][:]
            ged4 = h5file['seq_7'][:]
            stage = h5file['label'][()]

        def to_image(data):
            tensor = torch.tensor(data[None, :, :, :], dtype=torch.float32)  # (1, D, H, W)
            return tio.Image(tensor=tensor, type=tio.INTENSITY)
        
        subject = tio.Subject(
            t1=to_image(t1),
            t2=to_image(t2),
            dwi=to_image(dwi),
            ged1=to_image(ged1),
            ged2=to_image(ged2),
            ged3=to_image(ged3),
            ged4=to_image(ged4)
        )
        if self.transform:
            subject = self.transform(subject)
        
        # ===== 标签转换 =====
        # Subtask1: S1–S3 vs. S4 → 0 vs 1
        label_subtask1 = 1 if stage == 4 else 0

        # Subtask2: S1 vs. S2–S4 → 1 vs 0
        label_subtask2 = 1 if stage == 1 else 0

        return {
            't1': subject['t1'].data,
            't2': subject['t2'].data,
            'dwi': subject['dwi'].data,
            'ged1': subject['ged1'].data,
            'ged2': subject['ged2'].data,
            'ged3': subject['ged3'].data,
            'ged4': subject['ged4'].data,
            'label_subtask1': torch.tensor(label_subtask1, dtype=torch.float32),
            'label_subtask2': torch.tensor(label_subtask2, dtype=torch.float32),
            'filename': filename
        }
    
    def get_labels(self, label_name):
        """
        获取所有样本的指定子任务标签，返回 numpy 数组
        label_name: 'label_subtask1' 或 'label_subtask2'
        """
        labels = []
        for file_path in self.h5_files:
            with h5py.File(file_path, 'r') as h5file:
                stage = h5file['label'][()]
            if label_name == 'label_subtask1':
                label = 1 if stage == 4 else 0
            elif label_name == 'label_subtask2':
                label = 1 if stage == 1 else 0
            else:
                raise ValueError("label_name must be 'label_subtask1' or 'label_subtask2'")
            labels.append(label)
        return np.array(labels)
