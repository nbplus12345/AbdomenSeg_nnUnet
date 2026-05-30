import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config_utils import get_args, load_config


def resolve_config_path(config_path: str) -> Path:
    """将配置文件路径转换为绝对路径。"""
    path = Path(config_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_project_path(path: str) -> Path:
    """将项目内相对路径转换为绝对路径。"""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_nii_files(folder: Path) -> list[Path]:
    """获取指定文件夹下所有 .nii.gz 文件，并按文件名排序。"""
    return sorted(folder.glob("*.nii.gz"))


def extract_case_id(filename: str) -> str | None:
    """从文件名中提取病例编号。"""
    match = re.search(r"(\d+)", filename)

    if match is None:
        return None

    return match.group(1)


def load_nifti_shape(path: Path) -> tuple[int, ...]:
    """读取 NIfTI 文件的空间尺寸。"""
    nifti_img = nib.load(str(path))
    return nifti_img.shape


def get_unique_labels(path: Path) -> np.ndarray:
    """读取标签文件中实际出现的类别编号。"""
    label_img = nib.load(str(path))
    label_data = np.asanyarray(label_img.dataobj)
    unique_labels = np.unique(label_data).astype(int)

    return unique_labels


def main() -> None:
    """检查 BTCV 原始数据是否满足后续转换要求。"""
    args = get_args()
    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))

    images_dir = resolve_project_path(config.paths.raw_images_dir)
    labels_dir = resolve_project_path(config.paths.raw_labels_dir)

    expected_num_cases = config.dataset.expected_num_cases
    expected_label_min = config.dataset.expected_label_min
    expected_label_max = config.dataset.expected_label_max

    print("[INFO] BTCV raw dataset check started.")
    print(f"[INFO] Images dir: {images_dir}")
    print(f"[INFO] Labels dir: {labels_dir}")

    if not images_dir.exists():
        print(f"[ERROR] Images folder does not exist: {images_dir}")
        sys.exit(1)

    if not labels_dir.exists():
        print(f"[ERROR] Labels folder does not exist: {labels_dir}")
        sys.exit(1)

    image_files = get_nii_files(images_dir)
    label_files = get_nii_files(labels_dir)

    print(f"[INFO] Found image files: {len(image_files)}")
    print(f"[INFO] Found label files: {len(label_files)}")

    if len(image_files) != expected_num_cases:
        print(
            f"[WARNING] Expected {expected_num_cases} images, but found {len(image_files)}."
        )

    if len(label_files) != expected_num_cases:
        print(
            f"[WARNING] Expected {expected_num_cases} labels, but found {len(label_files)}."
        )

    image_map = {}
    label_map = {}

    # 建立 image case_id -> image path 的映射
    for image_path in image_files:
        case_id = extract_case_id(image_path.name)

        if case_id is None:
            print(f"[ERROR] Cannot extract case id from image file: {image_path.name}")
            sys.exit(1)

        image_map[case_id] = image_path

    # 建立 label case_id -> label path 的映射
    for label_path in label_files:
        case_id = extract_case_id(label_path.name)

        if case_id is None:
            print(f"[ERROR] Cannot extract case id from label file: {label_path.name}")
            sys.exit(1)

        label_map[case_id] = label_path

    image_ids = set(image_map.keys())
    label_ids = set(label_map.keys())

    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)

    if missing_labels:
        print(f"[ERROR] These image cases have no matching label: {missing_labels}")
        sys.exit(1)

    if missing_images:
        print(f"[ERROR] These label cases have no matching image: {missing_images}")
        sys.exit(1)

    common_ids = sorted(image_ids & label_ids)

    print(f"[INFO] Matched cases: {len(common_ids)}")
    print("[INFO] Checking each matched case...")

    all_label_values = set()

    # 逐个病例检查图像和标签是否一致
    for case_id in common_ids:
        image_path = image_map[case_id]
        label_path = label_map[case_id]

        image_shape = load_nifti_shape(image_path)
        label_shape = load_nifti_shape(label_path)

        if image_shape != label_shape:
            print(f"[ERROR] Shape mismatch in case {case_id}")
            print(f"[ERROR] Image: {image_path.name}, shape: {image_shape}")
            print(f"[ERROR] Label: {label_path.name}, shape: {label_shape}")
            sys.exit(1)

        unique_labels = get_unique_labels(label_path)
        all_label_values.update(unique_labels.tolist())

        if (
            unique_labels.min() < expected_label_min
            or unique_labels.max() > expected_label_max
        ):
            print(f"[ERROR] Label values out of expected range in case {case_id}")
            print(f"[ERROR] Label file: {label_path.name}")
            print(f"[ERROR] Unique labels: {unique_labels.tolist()}")
            sys.exit(1)

        if np.array_equal(unique_labels, np.array([0, 1])):
            print(f"[WARNING] Case {case_id} has only binary labels [0, 1].")
            print("[WARNING] This may not be the BTCV multi-organ label you want.")

        print(
            f"[INFO] Case {case_id}: "
            f"image={image_path.name}, label={label_path.name}, "
            f"shape={image_shape}, labels={unique_labels.tolist()}"
        )

    all_label_values = sorted(all_label_values)

    print(f"[INFO] Global label values found in all cases: {all_label_values}")

    if all_label_values == [0, 1]:
        print("[ERROR] All labels are binary [0, 1].")
        print("[ERROR] This is probably NOT the BTCV 13-organ multi-class label set.")
        sys.exit(1)

    expected_values = list(range(expected_label_min, expected_label_max + 1))
    missing_values = sorted(set(expected_values) - set(all_label_values))

    if missing_values:
        print(
            f"[WARNING] Some labels from 0-13 were not found globally: {missing_values}"
        )

    print("[SUCCESS] BTCV raw dataset check finished.")
    print("[SUCCESS] The dataset looks ready for nnU-Net format conversion.")


if __name__ == "__main__":
    main()
