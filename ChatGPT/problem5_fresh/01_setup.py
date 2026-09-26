import os, sys, json, time, random, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from IPython.display import display
assert sys.version_info >= (3, 10)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
device = torch.device("cpu")
n_epochs = 20
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
versions = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__, "device": str(device)}
print(versions)
