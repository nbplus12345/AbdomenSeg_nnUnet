import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config_utils import get_args, load_config  # noqa: E402


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


def setup_nnunet_env(config) -> tuple[Path, Path, Path]:
    """根据项目配置设置 nnU-Net v2 所需环境变量。"""
    nnunet_raw = resolve_project_path(config.paths.nnunet_raw)
    nnunet_preprocessed = resolve_project_path(config.paths.nnunet_preprocessed)
    nnunet_results = resolve_project_path(config.paths.nnunet_results)

    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)

    return nnunet_raw, nnunet_preprocessed, nnunet_results


def get_nii_files(folder: Path) -> list[Path]:
    """获取指定文件夹下所有 .nii.gz 文件，并按文件名排序。"""
    return sorted(folder.glob("*.nii.gz"))


def extract_case_id(filename: str) -> str | None:
    """从文件名中提取病例编号。"""
    match = re.search(r"(\d+)", filename)

    if match is None:
        return None

    return match.group(1)


def build_case_map(files: list[Path]) -> dict[str, Path]:
    """根据病例编号建立 case_id 到文件路径的映射。"""
    case_map = {}

    for file_path in files:
        case_id = extract_case_id(file_path.name)

        if case_id is None:
            print(f"[ERROR] Cannot extract case id from file: {file_path.name}")
            sys.exit(1)

        if case_id in case_map:
            print(f"[ERROR] Duplicate case id found: {case_id}")
            print(f"[ERROR] First file: {case_map[case_id]}")
            print(f"[ERROR] Second file: {file_path}")
            sys.exit(1)

        case_map[case_id] = file_path

    return case_map


def split_train_test_cases(
    case_ids: list[str],
    test_size: int,
    random_seed: int,
) -> tuple[list[str], list[str]]:
    """使用固定随机种子划分训练集和测试集。"""
    if test_size <= 0:
        return case_ids, []

    if test_size >= len(case_ids):
        print("[ERROR] test_size must be smaller than total number of cases.")
        sys.exit(1)

    random_generator = random.Random(random_seed)
    shuffled_case_ids = case_ids.copy()
    random_generator.shuffle(shuffled_case_ids)

    test_case_ids = sorted(shuffled_case_ids[:test_size])
    train_case_ids = sorted(shuffled_case_ids[test_size:])

    return train_case_ids, test_case_ids


def make_dataset_json(
    output_dataset_dir: Path,
    labels: dict,
    num_training: int,
) -> None:
    """生成 nnU-Net v2 所需的 dataset.json 文件。"""
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }

    json_path = output_dataset_dir / "dataset.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(dataset_json, file, indent=4, ensure_ascii=False)

    print(f"[INFO] dataset.json saved to: {json_path}")


def main() -> None:
    """将 BTCV 原始数据转换为 nnU-Net v2 标准格式。"""
    args = get_args()
    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))

    nnunet_raw_dir, _, _ = setup_nnunet_env(config)

    raw_images_dir = resolve_project_path(config.paths.raw_images_dir)
    raw_labels_dir = resolve_project_path(config.paths.raw_labels_dir)

    dataset_id = config.project.dataset_id
    dataset_name = config.project.dataset_name
    nnunet_dataset_name = f"Dataset{dataset_id}_{dataset_name}"

    output_dataset_dir = nnunet_raw_dir / nnunet_dataset_name
    images_tr_dir = output_dataset_dir / "imagesTr"
    labels_tr_dir = output_dataset_dir / "labelsTr"
    images_ts_dir = output_dataset_dir / "imagesTs"
    labels_ts_dir = output_dataset_dir / "labelsTs"

    overwrite = bool(config.conversion.overwrite) if config.conversion else False
    test_size = int(config.conversion.test_size) if config.conversion.test_size else 5
    random_seed = (
        int(config.conversion.random_seed) if config.conversion.random_seed else 42
    )

    print("[INFO] Convert BTCV to nnU-Net v2 format started.")
    print(f"[INFO] Raw images dir: {raw_images_dir}")
    print(f"[INFO] Raw labels dir: {raw_labels_dir}")
    print(f"[INFO] Output dataset dir: {output_dataset_dir}")
    print(f"[INFO] Test size: {test_size}")
    print(f"[INFO] Random seed: {random_seed}")

    if not raw_images_dir.exists():
        print(f"[ERROR] Raw images folder does not exist: {raw_images_dir}")
        sys.exit(1)

    if not raw_labels_dir.exists():
        print(f"[ERROR] Raw labels folder does not exist: {raw_labels_dir}")
        sys.exit(1)

    image_files = get_nii_files(raw_images_dir)
    label_files = get_nii_files(raw_labels_dir)

    print(f"[INFO] Found image files: {len(image_files)}")
    print(f"[INFO] Found label files: {len(label_files)}")

    image_map = build_case_map(image_files)
    label_map = build_case_map(label_files)

    image_ids = set(image_map.keys())
    label_ids = set(label_map.keys())

    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)

    if missing_labels:
        print(f"[ERROR] Missing labels for image cases: {missing_labels}")
        sys.exit(1)

    if missing_images:
        print(f"[ERROR] Missing images for label cases: {missing_images}")
        sys.exit(1)

    case_ids = sorted(image_ids)

    print(f"[INFO] Matched cases: {len(case_ids)}")

    train_case_ids, test_case_ids = split_train_test_cases(
        case_ids=case_ids,
        test_size=test_size,
        random_seed=random_seed,
    )

    print(f"[INFO] Training cases: {len(train_case_ids)}")
    print(f"[INFO] Test cases: {len(test_case_ids)}")
    print(f"[INFO] Test case ids: {test_case_ids}")

    if output_dataset_dir.exists():
        if overwrite:
            print(
                f"[WARNING] Existing dataset folder will be removed: {output_dataset_dir}"
            )
            shutil.rmtree(output_dataset_dir)
        else:
            print(f"[ERROR] Output dataset folder already exists: {output_dataset_dir}")
            print(
                "[ERROR] Set conversion.overwrite to true if you want to regenerate it."
            )
            sys.exit(1)

    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)
    images_ts_dir.mkdir(parents=True, exist_ok=True)
    labels_ts_dir.mkdir(parents=True, exist_ok=True)

    # 复制训练集
    for case_id in train_case_ids:
        src_image = image_map[case_id]
        src_label = label_map[case_id]

        new_case_id = f"BTCV_{case_id}"

        dst_image = images_tr_dir / f"{new_case_id}_0000.nii.gz"
        dst_label = labels_tr_dir / f"{new_case_id}.nii.gz"

        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)

        print(f"[INFO] Train image copied: {src_image.name} -> {dst_image.name}")
        print(f"[INFO] Train label copied: {src_label.name} -> {dst_label.name}")

    # 复制测试集
    for case_id in test_case_ids:
        src_image = image_map[case_id]
        src_label = label_map[case_id]

        new_case_id = f"BTCV_{case_id}"

        dst_image = images_ts_dir / f"{new_case_id}_0000.nii.gz"
        dst_label = labels_ts_dir / f"{new_case_id}.nii.gz"

        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)

        print(f"[INFO] Test image copied: {src_image.name} -> {dst_image.name}")
        print(f"[INFO] Test label copied: {src_label.name} -> {dst_label.name}")

    make_dataset_json(
        output_dataset_dir=output_dataset_dir,
        labels=dict(config.labels),
        num_training=len(train_case_ids),
    )

    print("[SUCCESS] BTCV has been converted to nnU-Net v2 format.")
    print(f"[INFO] Dataset ID: {dataset_id}")
    print(f"[INFO] Dataset name: {nnunet_dataset_name}")
    print(f"[INFO] Output: {output_dataset_dir}")
    print(f"[INFO] Final training cases: {len(train_case_ids)}")
    print(f"[INFO] Final test cases: {len(test_case_ids)}")


if __name__ == "__main__":
    main()
