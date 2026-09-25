# Script 01 - Setup: imports, version checks, plotting defaults, device selection
# Based on the "Setup" section of 10_neural_nets_with_pytorch.ipynb (Geron, 2025)
import sys
from packaging.version import Version

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

assert sys.version_info >= (3, 10)
# The book asks for PyTorch >= 2.6.0; this code only uses APIs available in 2.x,
# so we warn instead of failing on older installs.
if Version(torch.__version__) < Version("2.6.0"):
    print(f"Note: PyTorch {torch.__version__} is older than the book's 2.6.0")

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("PyTorch:", torch.__version__, "| device:", device)
