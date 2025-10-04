#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-file reproduction-style experiments for
'Comparison of novelty detection methods for multispectral images in rover-based planetary exploration missions'
Kerner et al., 2020.

Implements:
  - RX detector (spectral RX)
  - PCA reconstruction (spectral)
  - Conv Autoencoder (MSE; optional SSIM)
  - (Optional) Light BiGAN-style baseline (for completeness; disable to save time)

Training & hyperparameter selection are done on TYPICAL-ONLY data (train + val).
Test set mixes typical + novel for final evaluation (ROC/AUC).

Usage:
  python mastcam_novelty_singlefile.py --data_root /path/to/DATA_ROOT \
      --run_rx --run_pca --run_ae --ae_ssim 0 --run_bigan 0

Dependencies:
  numpy, scipy, scikit-learn, matplotlib (optional), pillow, imageio, scikit-image, torch, torchvision
"""

import os
import sys
import glob
import time
import math
import json
import argparse
import warnings
from typing import Tuple, List

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# IO & image
from PIL import Image
import imageio.v2 as imageio

# Metrics
from sklearn.metrics import roc_auc_score, roc_curve

# Learning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

# Torch for AE / (optional) BiGAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# SSIM (optional)
from skimage.metrics import structural_similarity as ssim

# ------------------------------
# Utils & helpers
# ------------------------------

def set_seed(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_files(d, exts=(".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    if not os.path.isdir(d):
        return []
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(d, f"*{ext}")))
    return sorted(files)


def load_tile(path: str) -> np.ndarray:
    """
    Load a single tile as float32 [0,1], shape (H,W,C).
    Supports .npy or image files (channels last).
    """
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:  # assume 0..255
            arr = arr / 255.0
        # ensure channels-last
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected .npy shape {arr.shape} for {path}")
        return arr
    else:
        img = imageio.imread(path)
        img = np.asarray(img)
        if img.ndim == 2:
            img = img[..., None]
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return img


def load_dir_as_array(d: str, limit=None) -> np.ndarray:
    files = list_files(d)
    if limit is not None and len(files) > limit:
        files = files[:limit]
    Xs = []
    for f in files:
        Xs.append(load_tile(f))
    if not Xs:
        return np.zeros((0, 64, 64, 1), dtype=np.float32)
    # ensure consistent shapes
    H, W, C = Xs[0].shape
    for i, x in enumerate(Xs):
        if x.shape != (H, W, C):
            raise ValueError(f"All tiles in {d} must share same shape: first={Xs[0].shape}, got {x.shape} at {i}:{files[i]}")
    return np.stack(Xs, axis=0).astype(np.float32)


def fit_band_standardizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=(0,1,2), keepdims=True)
    sigma = X_train.std(axis=(0,1,2), keepdims=True) + 1e-8
    return mu.astype(np.float32), sigma.astype(np.float32)

def apply_band_standardizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sigma
    # robust clip
    return np.clip(Z, -20.0, 20.0).astype(np.float32)


def manual_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(np.float64)
    finite = np.isfinite(scores)
    y_true = y_true[finite]; scores = scores[finite]
    # anomalies should be positive = 1 for roc_auc_score
    y_bin = (y_true == 1).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    try:
        return float(roc_auc_score(y_bin, scores))
    except Exception:
        return float("nan")


# ------------------------------
# Dataset wrapper for torch
# ------------------------------

class TilesDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X.transpose(0,3,1,2)  # to NCHW
        self.X = self.X.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])


# ------------------------------
# RX Detector (spectral)
# ------------------------------

def fit_rx_background(X_train: np.ndarray):
    """
    Fit RX background on typical-only training tiles.
    We gather all per-pixel spectral vectors (C-dim) to estimate mean and covariance.
    """
    N, H, W, C = X_train.shape
    Xv = X_train.reshape(-1, C)
    mu = Xv.mean(axis=0, keepdims=True)
    cov = EmpiricalCovariance().fit(Xv).covariance_
    # Regularize in case of ill-conditioning
    eps = 1e-6
    cov = cov + eps * np.eye(C, dtype=cov.dtype)
    cov_inv = np.linalg.inv(cov)
    return mu.astype(np.float64), cov_inv.astype(np.float64)

def rx_score_tiles(X: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray, agg: str="mean") -> np.ndarray:
    """
    Compute RX for each pixel, then aggregate to a tile-level anomaly score.
    agg in {'mean','max','p95'}
    """
    N, H, W, C = X.shape
    Xc = X.reshape(-1, C) - mu  # (N*H*W, C)
    # Mahalanobis: (x-μ)^T Σ^{-1} (x-μ)
    m = np.einsum("nc,cd,nd->n", Xc, cov_inv, Xc, optimize=True)
    m = m.reshape(N, H, W)
    if agg == "mean":
        return m.mean(axis=(1,2))
    elif agg == "max":
        return m.max(axis=(1,2))
    elif agg == "p95":
        return np.percentile(m, 95, axis=(1,2))
    else:
        raise ValueError("Unknown agg for RX")


# ------------------------------
# PCA Reconstruction (spectral)
# ------------------------------

def fit_pca_spectral(X_train: np.ndarray, n_components: int = 5, whiten: bool = False, random_state: int = 123):
    """
    Fit PCA over per-pixel spectral vectors (C-dim).
    """
    N, H, W, C = X_train.shape
    Xv = X_train.reshape(-1, C)
    pca = PCA(n_components=min(n_components, C), whiten=whiten, svd_solver="auto", random_state=random_state)
    pca.fit(Xv)
    return pca

def pca_recon_error_tiles(X: np.ndarray, pca: PCA, agg: str="mean") -> np.ndarray:
    """
    Reconstruction MSE per pixel, aggregated to tile-level.
    """
    N, H, W, C = X.shape
    Xv = X.reshape(-1, C)
    Z  = pca.transform(Xv)
    Xr = pca.inverse_transform(Z)
    err = ((Xv - Xr) ** 2).mean(axis=1)
    err = err.reshape(N, H, W)
    if agg == "mean":
        return err.mean(axis=(1,2))
    elif agg == "max":
        return err.max(axis=(1,2))
    elif agg == "p95":
        return np.percentile(err, 95, axis=(1,2))
    else:
        raise ValueError("Unknown agg for PCA error")


# ------------------------------
# Convolutional Autoencoder (MSE / optional SSIM)
# ------------------------------

class ConvAE(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(in_ch, 32, 3, padding=1)   # H,W
        self.e2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # H/2
        self.e3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # H/4
        self.e4 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # H/8
        # Decoder
        self.d1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # H/4
        self.d2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # H/2
        self.d3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # H
        self.out = nn.Conv2d(32, in_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = F.relu(self.e3(x))
        x = F.relu(self.e4(x))
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        x = torch.sigmoid(self.out(x))
        return x


def train_autoencoder(X_train: np.ndarray,
                      X_val: np.ndarray,
                      max_epochs=100,
                      batch_size=64,
                      lr=1e-3,
                      patience=10,
                      device="cuda" if torch.cuda.is_available() else "cpu",
                      use_ssim_loss: bool=False):
    """
    Train conv AE on typical-only train/val.
    If use_ssim_loss=True, we minimize 1-SSIM + small MSE for stability.
    """
    model = ConvAE(in_ch=X_train.shape[3]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    wait = 0

    train_loader = DataLoader(TilesDataset(X_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(TilesDataset(X_val),   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    def batch_loss(x, y):
        if use_ssim_loss:
            # 1 - SSIM over channels averaged + small MSE
            # Convert to CHW numpy per-sample; SSIM is CPU-heavy -> loop
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            ssim_vals = []
            for i in range(x_np.shape[0]):
                # compute mean SSIM across channels
                s_per_c = []
                for c in range(x_np.shape[1]):
                    s_per_c.append(ssim(y_np[i, c], x_np[i, c], data_range=1.0))
                ssim_vals.append(np.mean(s_per_c))
            ssim_term = 1.0 - np.mean(ssim_vals)
            mse_term  = F.mse_loss(x, y).item()
            return ssim_term + 0.1 * mse_term
        else:
            return F.mse_loss(x, y)

    for epoch in range(1, max_epochs+1):
        model.train()
        tr_loss = 0.0
        for xb in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yb = model(xb)
            loss = F.mse_loss(yb, xb) if not use_ssim_loss else None
            if use_ssim_loss:
                # compute hybrid loss (slow on CPU)
                recon = yb
                loss_val = batch_loss(recon, xb)
                loss = torch.as_tensor(loss_val, dtype=torch.float32, device=device)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = model(xb)
                if use_ssim_loss:
                    # compute hybrid validation loss on CPU
                    loss_val = batch_loss(yb, xb)
                    loss = torch.as_tensor(loss_val, dtype=torch.float32, device=device)
                else:
                    loss = F.mse_loss(yb, xb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

        print(f"[AE] epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

        if va_loss + 1e-8 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[AE] Early stopping at epoch {epoch} (best val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def ae_scores_on_tiles(model: nn.Module,
                       X: np.ndarray,
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       agg="mean",
                       use_ssim_score: bool=False) -> np.ndarray:
    """
    Compute tile-level anomaly scores via recon error (MSE or 1-SSIM).
    """
    model.eval()
    loader = DataLoader(TilesDataset(X), batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    scores = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = model(xb)
            if use_ssim_score:
                # per-sample 1-SSIM averaged across channels
                xb_np = xb.detach().cpu().numpy()
                yb_np = yb.detach().cpu().numpy()
                batch_scores = []
                for i in range(xb_np.shape[0]):
                    s_per_c = []
                    for c in range(xb_np.shape[1]):
                        s_per_c.append(ssim(xb_np[i, c], yb_np[i, c], data_range=1.0))
                    batch_scores.append(1.0 - float(np.mean(s_per_c)))
                scores.extend(batch_scores)
            else:
                err = F.mse_loss(yb, xb, reduction="none")  # (N,C,H,W)
                # MSE over C,H,W per sample
                err = err.mean(dim=(1,2,3)).detach().cpu().numpy()
                scores.extend(err.tolist())

    return np.asarray(scores, dtype=np.float64)


# ------------------------------
# (Optional) Light BiGAN Baseline
# ------------------------------

class SimpleEncoder(nn.Module):
    def __init__(self, in_ch, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512* (4*4), z_dim)  # assumes divisible by 16; adapt if needed

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        z = self.fc(h)
        return z

class SimpleDecoder(nn.Module):
    def __init__(self, out_ch, z_dim=64, base=512):
        super().__init__()
        self.fc = nn.Linear(z_dim, base*(4*4))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_ch, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        x = self.net(h)
        return x

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_ch, z_dim=64):
        super().__init__()
        # Judge joint (x,z). Concatenate z to feature map by channel replication after projecting.
        self.x_feat = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.z_proj = nn.Linear(z_dim, 256*4*4)
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4),  # 1×1
        )

    def forward(self, x, z):
        fx = self.x_feat(x)
        zmap = self.z_proj(z).view(z.size(0), 256, 4, 4)
        cat = torch.cat([fx, zmap], dim=1)
        out = self.head(cat).view(-1)
        return out

def train_bigan_light(X_train, X_val, epochs=20, batch_size=64, lr=2e-4, z_dim=64,
                      device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Lightweight BiGAN-like training (normal-only). Uses WGAN-GP-ish losses (without full GP to keep it short).
    This is included for completeness; it won't exactly match paper results but serves as a comparable baseline.
    """
    N, H, W, C = X_train.shape
    assert H % 16 == 0 and W % 16 == 0, "For the light BiGAN we assume dims divisible by 16 (e.g., 64×64)."

    G_e = SimpleEncoder(C, z_dim).to(device)
    G_d = SimpleDecoder(C, z_dim).to(device)
    D   = SimpleDiscriminator(C, z_dim).to(device)

    opt_G = torch.optim.Adam(list(G_e.parameters()) + list(G_d.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    train_loader = DataLoader(TilesDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TilesDataset(X_val), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        G_e.train(); G_d.train(); D.train()
        for xb in train_loader:
            xb = xb.to(device)
            b  = xb.size(0)
            # Encode real to z
            z_real = G_e(xb)
            # Reconstruct
            xr = G_d(z_real).detach()  # stop G on D step

            # Sample prior
            z_prior = torch.randn(b, z_real.size(1), device=device)

            # --- Train D: maximize D(x,z_real) + D(G(z), z)
            opt_D.zero_grad(set_to_none=True)
            d_real = D(xb, z_real)
            x_fake = G_d(z_prior).detach()
            d_fake = D(x_fake, z_prior)
            loss_D = -(d_real.mean() + d_fake.mean())
            loss_D.backward()
            opt_D.step()

            # --- Train G/E: minimize -D(x,z_real) - D(G(z), z) + small recon
            opt_G.zero_grad(set_to_none=True)
            z_real = G_e(xb)
            x_rec  = G_d(z_real)
            d_real = D(xb, z_real)
            d_fake = D(G_d(z_prior), z_prior)
            recon  = F.mse_loss(x_rec, xb)
            loss_G = - (d_real.mean() + d_fake.mean()) + 10.0 * recon
            loss_G.backward()
            opt_G.step()

        # crude val: recon MSE on val
        G_e.eval(); G_d.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                xr = G_d(G_e(xb))
                vloss += F.mse_loss(xr, xb, reduction="sum").item()
        vloss /= len(val_loader.dataset)
        print(f"[BiGAN] epoch {ep:03d} val_recon={vloss:.6f}")

        if vloss + 1e-8 < best_val:
            best_val = vloss
            best_state = {
                "G_e": {k: v.detach().cpu().clone() for k,v in G_e.state_dict().items()},
                "G_d": {k: v.detach().cpu().clone() for k,v in G_d.state_dict().items()},
                "D":   {k: v.detach().cpu().clone() for k,v in D.state_dict().items()},
            }

    if best_state is not None:
        G_e.load_state_dict(best_state["G_e"])
        G_d.load_state_dict(best_state["G_d"])
        D.load_state_dict(best_state["D"])
    return G_e, G_d, D

def bigan_scores_on_tiles(G_e, G_d, X, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Use reconstruction error in pixel space as anomaly score (other choices possible).
    """
    loader = DataLoader(TilesDataset(X), batch_size=64, shuffle=False)
    scores = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            xr = G_d(G_e(xb))
            err = F.mse_loss(xr, xb, reduction="none").mean(dim=(1,2,3)).detach().cpu().numpy()
            scores.extend(err.tolist())
    return np.asarray(scores, dtype=np.float64)


# ------------------------------
# Experiment runner
# ------------------------------

def evaluate_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """
    Given anomaly scores for positive class (novel) and negative (typical),
    compute ROC AUC (anomaly-positive).
    """
    y = np.concatenate([np.ones_like(scores_pos, int), np.zeros_like(scores_neg, int)], axis=0)
    s = np.concatenate([scores_pos, scores_neg], axis=0)
    return manual_auc(y, s)


# ------------------------------
# Experiment runner
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_rx", type=int, default=1)
    ap.add_argument("--run_pca", type=int, default=1)
    ap.add_argument("--run_ae", type=int, default=1)
    ap.add_argument("--run_bigan", type=int, default=0)   # off by default
    ap.add_argument("--rx_agg", type=str, default="mean", choices=["mean","max","p95"])
    ap.add_argument("--pca_k", type=int, default=5)
    ap.add_argument("--pca_whiten", type=int, default=0)
    ap.add_argument("--ae_epochs", type=int, default=100)
    ap.add_argument("--ae_batch", type=int, default=64)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--ae_patience", type=int, default=10)
    ap.add_argument("--ae_ssim", type=int, default=0)     # 1 to use SSIM-based loss and/or score
    ap.add_argument("--bigan_epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)

    # Fixed directories
    train_data        = "Data/Full/train_typical"
    validation_data   = "Data/Full/validation_typical"
    test_typical_data = "Data/Full/test_typical"
    test_anomaly_data = "Data/Full/test_novel/all"

    print("[IO] Loading data ...")
    Xtr = load_dir_as_array(train_data)
    Xva = load_dir_as_array(validation_data)
    Xt0 = load_dir_as_array(test_typical_data)
    Xt1 = load_dir_as_array(test_anomaly_data)

    if Xtr.size == 0 or Xva.size == 0 or Xt0.size == 0 or Xt1.size == 0:
        raise RuntimeError("One or more required folders are empty or missing tiles. "
                           "Expect: Data/Full/train_typical, Data/Full/validation_typical, "
                           "Data/Full/test_typical, Data/Full/test_novel/all")

    # Sanity: consistent shapes & channels
    H, W, C = Xtr.shape[1:]
    for name, X in [("val", Xva), ("test_typical", Xt0), ("test_novel", Xt1)]:
        if X.shape[1:] != (H,W,C):
            raise ValueError(f"All splits must share same (H,W,C). Train has {(H,W,C)}, but {name} has {X.shape[1:]}")

    print(f"[shape] H={H}, W={W}, C={C}")
    print(f"[counts] train={Xtr.shape[0]} | val={Xva.shape[0]} | test_typ={Xt0.shape[0]} | test_nov={Xt1.shape[0]}")

    # Standardize per band using TRAIN stats (fit on train, apply to all)
    mu, sig = fit_band_standardizer(Xtr)
    Xtr = apply_band_standardizer(Xtr, mu, sig)
    Xva = apply_band_standardizer(Xva, mu, sig)
    Xt0 = apply_band_standardizer(Xt0, mu, sig)
    Xt1 = apply_band_standardizer(Xt1, mu, sig)

    results = {}

    # ---------------- RX ----------------
    if args.run_rx:
        print("\n[RX] Fitting RX background (typical-only train)...")
        t0 = time.time()
        mu_rx, covinv_rx = fit_rx_background(Xtr)
        train_time = time.time() - t0
        print(f"[RX] Fit in {train_time:.2f}s")

        print("[RX] Scoring test tiles ...")
        s_typ = rx_score_tiles(Xt0, mu_rx, covinv_rx, agg=args.rx_agg)
        s_nov = rx_score_tiles(Xt1, mu_rx, covinv_rx, agg=args.rx_agg)
        auc = evaluate_auc(s_nov, s_typ)
        print(f"[RX] AUC (anomaly-positive): {auc:.4f}  (agg={args.rx_agg})")
        results["RX"] = {"AUC": auc, "agg": args.rx_agg}

    # ---------------- PCA ----------------
    if args.run_pca:
        print("\n[PCA] Fitting spectral PCA on per-pixel vectors (typical-only train)...")
        t0 = time.time()
        pca = fit_pca_spectral(Xtr, n_components=args.pca_k, whiten=bool(args.pca_whiten), random_state=args.seed)
        train_time = time.time() - t0
        print(f"[PCA] Fit in {train_time:.2f}s | k={pca.n_components_} (whiten={bool(args.pca_whiten)})")

        print("[PCA] Scoring test tiles ...")
        s_typ = pca_recon_error_tiles(Xt0, pca, agg="mean")
        s_nov = pca_recon_error_tiles(Xt1, pca, agg="mean")
        auc = evaluate_auc(s_nov, s_typ)
        print(f"[PCA] AUC (anomaly-positive): {auc:.4f}")
        results["PCA"] = {"AUC": auc, "k": int(pca.n_components_), "whiten": bool(args.pca_whiten)}

    # ---------------- AE ----------------
    if args.run_ae:
        print("\n[AE] Training convolutional autoencoder on typical-only train/val ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        ae = train_autoencoder(
            Xtr, Xva,
            max_epochs=args.ae_epochs,
            batch_size=args.ae_batch,
            lr=args.ae_lr,
            patience=args.ae_patience,
            device=device,
            use_ssim_loss=bool(args.ae_ssim)
        )
        train_time = time.time() - t0
        print(f"[AE] Done in {train_time/60.0:.1f} min")

        print("[AE] Scoring test tiles ...")
        s_typ = ae_scores_on_tiles(ae, Xt0, device=device, agg="mean", use_ssim_score=bool(args.ae_ssim))
        s_nov = ae_scores_on_tiles(ae, Xt1, device=device, agg="mean", use_ssim_score=bool(args.ae_ssim))
        auc = evaluate_auc(s_nov, s_typ)
        print(f"[AE] AUC (anomaly-positive): {auc:.4f}  (score={'1-SSIM' if args.ae_ssim else 'MSE'})")
        results["AE"] = {
            "AUC": auc,
            "loss": "1-SSIM+MSE" if args.ae_ssim else "MSE",
            "score": "1-SSIM" if args.ae_ssim else "MSE"
        }

    # ---------------- BiGAN (Optional) ----------------
    if args.run_bigan:
        print("\n[BiGAN] Training lightweight BiGAN-style baseline on typical-only train/val ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        G_e, G_d, D = train_bigan_light(
            Xtr, Xva,
            epochs=args.bigan_epochs,
            batch_size=64,
            lr=2e-4,
            z_dim=64,
            device=device
        )
        train_time = time.time() - t0
        print(f"[BiGAN] Done in {train_time/60.0:.1f} min")

        print("[BiGAN] Scoring test tiles ...")
        s_typ = bigan_scores_on_tiles(G_e, G_d, Xt0, device=device)
        s_nov = bigan_scores_on_tiles(G_e, G_d, Xt1, device=device)
        auc = evaluate_auc(s_nov, s_typ)
        print(f"[BiGAN] AUC (anomaly-positive): {auc:.4f}")
        results["BiGAN"] = {"AUC": auc}

    # Summary
    print("\n======= RESULTS (anomaly-positive ROC AUC) =======")
    for k, v in results.items():
        print(f"{k:6s} : {v['AUC']:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
