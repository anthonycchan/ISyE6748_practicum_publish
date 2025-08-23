import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
import torch

# Make sure backend is PyTorch
tl.set_backend("pytorch")

# Pick device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch sees CUDA:", torch.cuda.is_available())
print("DEVICE chosen:", DEVICE)

# Create a random tensor directly on DEVICE
X = torch.randn(30, 30, 30, device=DEVICE)

# Run a small CP decomposition
weights, factors = parafac(X, rank=3, init="random", n_iter_max=5)

# Check where the results live
print("weights.device ->", weights.device)
print("factors.device ->", [f.device for f in factors])
