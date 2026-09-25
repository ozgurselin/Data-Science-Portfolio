"""Stage 1: imports and reproducible configuration (shared notebook namespace)."""
from pathlib import Path
import copy
import csv
import json
import platform
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.transforms import v2 as T

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.1
# CPU is fast for this small MLP and makes this submitted run reproducible.
device = torch.device("cpu")
torch.set_num_threads(2)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True})
print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"Device: {device}; seed: {SEED}; epochs: {EPOCHS}; batch size: {BATCH_SIZE}")
