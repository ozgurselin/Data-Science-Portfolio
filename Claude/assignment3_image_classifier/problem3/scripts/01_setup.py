# Step 1 - Imports, version checks, plot defaults and device selection
import sys
import warnings
from packaging.version import Version

# torchvision.io's optional JPEG/PNG backend isn't used here; hide its load warning
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T

assert sys.version_info >= (3, 10)
if Version(torch.__version__) < Version("2.6.0"):
    print(f"Note: PyTorch {torch.__version__} is older than the book's 2.6.0; "
          "this section runs on any PyTorch 2.x")

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"torchmetrics {torchmetrics.__version__} | device: {device}")
