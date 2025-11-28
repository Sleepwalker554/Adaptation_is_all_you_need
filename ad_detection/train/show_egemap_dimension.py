from pathlib import Path

import torch

from config import PROJECT_ROOT


def main() -> None:
    """Load S001.egemaps.pt and print its shape."""
    egemap_path = PROJECT_ROOT / "data/processed/ADReSS_egemap_features/S001.egemaps.pt"

    if not egemap_path.exists():
        raise FileNotFoundError(f"Feature file not found: {egemap_path}")

    features = torch.load(egemap_path)
    print(f"Loaded {egemap_path}")
    print(f"Tensor shape: {tuple(features.shape)}")


if __name__ == "__main__":
    main()
