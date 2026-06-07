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

    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)

    if config.nnunet and config.nnunet.n_proc_DA:
        os.environ["nnUNet_n_proc_DA"] = str(config.nnunet.n_proc_DA)


def main() -> None:
    """运行 nnU-Net 官方 plan_and_preprocess。"""
    args = get_args()
    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))

    setup_nnunet_env(config)

    dataset_id = str(config.project.dataset_id)

    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        dataset_id,
    ]

    if config.preprocess.verify_dataset_integrity:
        command.append("--verify_dataset_integrity")

    print("[INFO] Running command:")
    print("[INFO] " + " ".join(command))
    print(f"[INFO] nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"[INFO] nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"[INFO] nnUNet_results: {os.environ['nnUNet_results']}")
    print(f"[INFO] nnUNet_n_proc_DA: {os.environ.get('nnUNet_n_proc_DA')}")

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
