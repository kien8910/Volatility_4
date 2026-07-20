from __future__ import annotations

import platform
from pathlib import Path

import psutil
import torch


REQUIRED_FILES = [
    "data/processed/step3_residual_state.parquet",
    "data/processed/step3_residual_targets.parquet",
    "results/step3/oos_p_predictions.parquet",
    "data/splits/split_manifest.csv",
    "data/splits/expanding_folds.csv",
    "configs/step4_static_graph.yaml",
]


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"GPU name: {props.name}")
        print(f"GPU memory GB: {props.total_memory / 1024**3:.2f}")
    else:
        print("GPU name: unavailable")
        print("GPU memory GB: unavailable")
    print(f"CPU count: {psutil.cpu_count(logical=True)}")
    print(f"RAM GB: {psutil.virtual_memory().total / 1024**3:.2f}")
    print("Required files:")
    for item in REQUIRED_FILES:
        print(f"  {item}: {'OK' if Path(item).exists() else 'MISSING'}")


if __name__ == "__main__":
    main()

