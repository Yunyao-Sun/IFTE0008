"""MVMD decomposition utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from config import CFG


def compact_mvmd(
    X: np.ndarray,
    K: int = None,
    alpha: float = None,
    tau: float = None,
    tol: float = None,
    max_iter: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compact MVMD implementation for ablation experiments.

    Parameters
    ----------
    X:
        Array with shape [time, channels].

    Returns
    -------
    modes:
        Array with shape [K, channels, time].
    omega:
        Center frequencies with shape [K].

    Note
    ----
    This implementation is intended for experimental ablation. For final publication-level
    results, validate decomposition output against a trusted MVMD implementation.
    """
    K = K or CFG.VMD_K
    alpha = alpha if alpha is not None else CFG.VMD_ALPHA
    tau = tau if tau is not None else CFG.VMD_TAU
    tol = tol if tol is not None else CFG.VMD_TOL
    max_iter = max_iter or CFG.VMD_MAX_ITER

    X = np.asarray(X, dtype=float)
    T, C = X.shape

    mu = np.nanmean(X, axis=0, keepdims=True)
    sigma = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    Xs = (X - mu) / sigma

    freqs = np.fft.fftshift(np.fft.fftfreq(T))
    X_hat = np.fft.fftshift(np.fft.fft(Xs, axis=0), axes=0)

    u_hat = np.zeros((K, T, C), dtype=complex)
    lambda_hat = np.zeros((T, C), dtype=complex)
    omega = np.linspace(0.0, 0.5, K + 2)[1:-1]
    pos = freqs >= 0

    for _ in range(max_iter):
        old = u_hat.copy()

        for k in range(K):
            residual = X_hat - (np.sum(u_hat, axis=0) - u_hat[k]) + lambda_hat / 2
            denominator = 1.0 + alpha * (freqs[:, None] - omega[k]) ** 2
            u_hat[k] = residual / denominator

            spectrum = np.sum(np.abs(u_hat[k, pos, :]) ** 2, axis=1)
            omega[k] = np.sum(freqs[pos] * spectrum) / (np.sum(spectrum) + 1e-12)

        lambda_hat += tau * (X_hat - np.sum(u_hat, axis=0))
        diff = np.linalg.norm(u_hat - old) / (np.linalg.norm(old) + 1e-12)
        if diff < tol:
            break

    modes = np.zeros((K, C, T), dtype=float)
    for k in range(K):
        modes[k] = np.fft.ifft(np.fft.ifftshift(u_hat[k], axes=0), axis=0).real.T

    return modes, omega


def modes_to_features(modes: np.ndarray, channel_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    K, C, T = modes.shape
    features = modes.transpose(2, 0, 1).reshape(T, K * C)

    names: List[str] = []
    for k in range(K):
        for channel in channel_names:
            names.append(f"mvmd_k{k + 1}_{channel}")

    return features, names


def global_mvmd_features(df: pd.DataFrame, channel_cols: List[str]) -> pd.DataFrame:
    modes, _ = compact_mvmd(df[channel_cols].values)
    features, names = modes_to_features(modes, channel_cols)
    return pd.DataFrame(features, index=df.index, columns=names)
