import os
import shutil
import SimpleITK as sitk


def affine_registration(fixed_image_path, moving_image_path):
    fixed = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed, moving)

    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return resampled


def batch_affine_register_and_save(input_folder, output_folder, fixed_name="T1.nii.gz"):
    os.makedirs(output_folder, exist_ok=True)

    failed_log_path = os.path.join(output_folder, "registration_failed.txt")
    with open(failed_log_path, "w") as fail_log:

        for case_name in os.listdir(input_folder):
            case_path = os.path.join(input_folder, case_name)
            if not os.path.isdir(case_path):
                continue

            output_case_path = os.path.join(output_folder, case_name)
            os.makedirs(output_case_path, exist_ok=True)

            fixed_path = os.path.join(case_path, fixed_name)

            for seq_file in os.listdir(case_path):
                seq_path = os.path.join(case_path, seq_file)
                output_path = os.path.join(output_case_path, seq_file)

                if seq_file == fixed_name:
                    # 复制T1
                    t1_file = "T1.nii.gz"
                    t1_path = os.path.join(case_path, t1_file)
                    if os.path.exists(t1_path):
                        shutil.copy(t1_path, os.path.join(output_case_path, t1_file))
                    shutil.copy(fixed_path, output_path)
                    continue

                try:
                    print(f"🔄 正在配准: {seq_file} → {fixed_name}")
                    registered_image = affine_registration(fixed_path, seq_path)
                    sitk.WriteImage(registered_image, output_path)
                except Exception as e:
                    print(f"❌ 配准失败: {seq_file}，原因：{str(e)}")
                    fail_log.write(f"{seq_path}\n")


def batch_affine_register_and_saveT1(input_folder, output_folder, fixed_name="T1.nii.gz"):
    """
    以 T1 为标准，只配准 T2/DWI，T1 直接复制。
    input_folder: Vendor_xxx 目录（里面是 case 目录）
    output_folder: 输出根目录（所有 vendor 的 case 会堆在一起）
    """
    os.makedirs(output_folder, exist_ok=True)
    failed_log_path = os.path.join(output_folder, "registration_failed.txt")

    with open(failed_log_path, "a", encoding="utf-8") as fail_log:
        for case_name in os.listdir(input_folder):
            case_path = os.path.join(input_folder, case_name)
            if not os.path.isdir(case_path):
                continue

            fixed_path = os.path.join(case_path, fixed_name)
            if not os.path.exists(fixed_path):
                fail_log.write(f"[SKIP] Missing T1: {fixed_path}\n")
                continue

            # ✅ 避免不同 vendor 有同名 case 覆盖：加上 vendor 名做前缀
            vendor_name = os.path.basename(input_folder)
            out_case_name = f"{vendor_name}_{case_name}"
            output_case_path = os.path.join(output_folder, out_case_name)
            os.makedirs(output_case_path, exist_ok=True)

            # 复制 T1
            shutil.copy(fixed_path, os.path.join(output_case_path, fixed_name))

            # 只配准 T2 / DWI（按你数据真实文件名改）
            for moving_name in ["T2.nii.gz", "DWI_800.nii.gz", "DWI.nii.gz"]:
                moving_path = os.path.join(case_path, moving_name)
                if not os.path.exists(moving_path):
                    continue
                try:
                    print(f"🔄 正在配准: {moving_name} → {fixed_name}")
                    registered_image = affine_registration(fixed_path, moving_path)
                    sitk.WriteImage(registered_image, os.path.join(output_case_path, moving_name))
                except Exception as e:
                    fail_log.write(f"[FAIL] {moving_path} | {e}\n")

if __name__ == "__main__":
    input_folder = r"E:\BaiduNetdiskDownload\LiQA_val\test1\Vendor_B2"
    output_folder = r"E:\BaiduNetdiskDownload\LiQA_val\Registered\Vendor_B2_registered"

    batch_affine_register_and_saveT1(input_folder, output_folder)