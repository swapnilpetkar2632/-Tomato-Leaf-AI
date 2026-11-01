"""
verify_env.py
-------------
STAGE 1 — Environment Setup Verification
Verifies installed packages and GPU/CPU device availability.
"""

import sys
from importlib.metadata import version

import flask
import torch
import torchvision
import numpy as np
import PIL
import sklearn
import tqdm

def main():
    print("=" * 70)
    print("  STAGE 1 -- Environment Setup Verification")
    print("=" * 70)

    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Flask Version     : {version('Flask')}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"Torchvision       : {torchvision.__version__}")
    print(f"NumPy Version     : {np.__version__}")
    print(f"Scikit-Learn      : {sklearn.__version__}")
    print(f"Pillow Version    : {PIL.__version__}")
    print(f"tqdm Version      : {tqdm.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-" * 70)
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    print(f"Device in Use     : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name          : {torch.cuda.get_device_name(0)}")
    else:
        print("Falling back to CPU for computation.")
    print("=" * 70)
    print("[OK] STAGE 1 Complete: Environment verified successfully.\n")

if __name__ == "__main__":
    main()
