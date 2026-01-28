import os
import nibabel as nib
import numpy as np
import h5py
from skimage.transform import resize

# 设置常量
TARGET_SHAPE = (224, 224, 32)
SEQUENCE_FILES = ["T1.nii.gz", "T2.nii.gz", "DWI_800.nii.gz"]

def load_nifti_as_array(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data

def z_score_normalize(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / std if std > 0 else volume - mean

def resize_volume(volume, target_shape=(224, 224, 32)):
    target_x, target_y, target_z = target_shape

    # Step 0: 确保 Z 在最后一维
    if volume.shape[0] == min(volume.shape):  # (Z, H, W)
        volume = np.transpose(volume, (1, 2, 0))  # → (H, W, Z)
    elif volume.shape[2] != min(volume.shape):  # Z 轴不在最后，强制转到最后
        print(f"[WARN] Unexpected shape: {volume.shape}, attempting to fix")
        volume = np.moveaxis(volume, -1, 2)  # 如果Z轴错位，强制移到 axis=2

    current_x, current_y, current_z = volume.shape

    # Step 1: Resize XY plane
    resized_slices = []
    for i in range(current_z):
        slice_2d = volume[:, :, i]
        resized_slice = resize(slice_2d, (target_x, target_y), mode='constant',
                               preserve_range=True, anti_aliasing=True)
        resized_slices.append(resized_slice)
    volume_xy_resized = np.stack(resized_slices, axis=2)

    # Step 2: Adjust Z slices
    if current_z > target_z:
        start = (current_z - target_z) // 2
        volume_final = volume_xy_resized[:, :, start:start + target_z]
        
    else:
        pad_before = (target_z - current_z) // 2
        pad_after = target_z - current_z - pad_before
        volume_final = np.pad(volume_xy_resized,
                              ((0, 0), (0, 0), (pad_before, pad_after)),
                              mode='constant', constant_values=0)

    return volume_final.astype(np.float32)

    
def create_h5_for_patient(patient_folder, output_folder, error_log):
    patient_id = os.path.basename(patient_folder)

    # try:
    #     label_str = patient_id.strip().split("-")[-1]
    #     label = int(label_str[-1])
    #     if label not in [1, 2, 3, 4]:
    #         raise ValueError(f"非法标签：{label}")
    # except Exception as e:
    #     error_log.write(f"{patient_id}: 标签错误 - {e}\n")
    #     return

    data_list = []
    label = np.zeros((1,), dtype=np.int32)  # 默认标签为0

    for seq_file in SEQUENCE_FILES:
        seq_path = os.path.join(patient_folder, seq_file)
        if not os.path.exists(seq_path):
            error_log.write(f"{patient_id}: 缺少序列 {seq_file}, 使用0填充\n")
            data_list.append(np.zeros(TARGET_SHAPE, dtype=np.float32))
            continue

        try:
            volume = load_nifti_as_array(seq_path)
            volume = resize_volume(volume, TARGET_SHAPE)
            volume = z_score_normalize(volume)
            data_list.append(volume)
        except Exception as e:
            error_log.write(f"{patient_id}: 处理序列 {seq_file} 失败 - {e}\n")
            data_list.append(np.zeros(TARGET_SHAPE, dtype=np.float32))  # 避免中断，仍然填0


    data_list.append(label)

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"{patient_id}.h5")
    with h5py.File(save_path, 'w') as f:
        for i, item in enumerate(data_list[:-1]):
            item = np.transpose(item, (2, 0, 1))  # (H, W, Z) → (Z, H, W)
            f.create_dataset(f"seq_{i+1}", data=item, compression="gzip")
        f.create_dataset("label", data=data_list[-1])
    print(f"✅ 已保存: {save_path}")

def batch_process(input_root, output_root, error_txt_path):
    with open(error_txt_path, 'w') as error_log:
        for folder in os.listdir(input_root):
            patient_folder = os.path.join(input_root, folder)
            if os.path.isdir(patient_folder):
                create_h5_for_patient(patient_folder, output_root, error_log)

# # === 用法设置 ===
# input_root = r'E:\BaiduNetdiskDownload\LiQA_val\Registered\Vendor_B2_registered'    # 原始输入目录
# output_root = r'E:\BaiduNetdiskDownload\LiQA_val\h5_output_dataset\Vendor_B2'           # H5 输出目录
# error_txt_path = r"E:\Wangsiqi2\CARE-liver\error_log.txt" # 异常记录文件
# if __name__ == "__main__":
#     batch_process(input_root, output_root, error_txt_path) 
