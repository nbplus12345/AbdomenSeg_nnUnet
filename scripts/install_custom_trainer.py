from pathlib import Path

import nnunetv2


def main() -> None:
    """向当前 nnU-Net v2 环境注册自定义 200 epoch trainer。"""
    nnunetv2_root = Path(nnunetv2.__file__).resolve().parent
    trainer_dir = nnunetv2_root / "training" / "nnUNetTrainer"

    trainer_file = trainer_dir / "nnUNetTrainer_200epochs.py"

    code = '''from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_200epochs(nnUNetTrainer):
    """自定义 nnU-Net trainer：最大训练轮数改为 200。"""

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=None,
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        self.num_epochs = 200
'''

    with open(trainer_file, "w", encoding="utf-8") as file:
        file.write(code)

    print(f"[SUCCESS] Custom trainer saved to: {trainer_file}")


if __name__ == "__main__":
    main()
