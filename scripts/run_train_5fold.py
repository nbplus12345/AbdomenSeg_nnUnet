import os
import subprocess
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


def setup_nnunet_env(config) -> None:
    """设置 nnU-Net 所需环境变量。"""
    nnunet_raw = resolve_project_path(config.paths.nnunet_raw)
    nnunet_preprocessed = resolve_project_path(config.paths.nnunet_preprocessed)
    nnunet_results = resolve_project_path(config.paths.nnunet_results)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)

    if config.nnunet and config.nnunet.n_proc_DA:
        os.environ["nnUNet_n_proc_DA"] = str(config.nnunet.n_proc_DA)


def main() -> None:
    """按顺序训练多个 fold。"""
    args = get_args()
    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))

    setup_nnunet_env(config)

    dataset_id = str(config.project.dataset_id)
    configuration = str(config.training.configuration)
    folds = list(config.training.folds)
    trainer = str(config.training.trainer)
    continue_training = bool(config.training.continue_training)

    print("[INFO] Start 5-fold training.")
    print(f"[INFO] Dataset ID: {dataset_id}")
    print(f"[INFO] Configuration: {configuration}")
    print(f"[INFO] Folds: {folds}")
    print(f"[INFO] Trainer: {trainer}")
    print(f"[INFO] Continue training: {continue_training}")
    print(f"[INFO] nnUNet_n_proc_DA: {os.environ.get('nnUNet_n_proc_DA')}")

    for fold in folds:
        command = [
            "nnUNetv2_train",
            dataset_id,
            configuration,
            str(fold),
            "-tr",
            trainer,
        ]

        if continue_training:
            command.append("--c")

        print("\n" + "=" * 80)
        print(f"[INFO] Start training fold {fold}")
        print("[INFO] Command: " + " ".join(command))
        print("=" * 80 + "\n")

        subprocess.run(command, check=True)

        print("\n" + "=" * 80)
        print(f"[SUCCESS] Fold {fold} finished.")
        print("=" * 80 + "\n")

    print("[SUCCESS] All folds finished.")


if __name__ == "__main__":
    main()
