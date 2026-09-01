"""
VaR / CVaR directly from an EMPIRICAL cost distribution (real data, no
simulation). Same definitions as before, just no Monte Carlo step feeding
into it anymore.
"""

import numpy as np
import config


def compute_var(costs: np.ndarray, confidence: float = None) -> float:
    confidence = confidence or config.VAR_CONFIDENCE
    return float(np.quantile(costs, confidence))


def compute_cvar(costs: np.ndarray, confidence: float = None) -> float:
    confidence = confidence or config.VAR_CONFIDENCE
    var = compute_var(costs, confidence)
    tail = costs[costs >= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def summarize_costs(costs: np.ndarray) -> dict:
    result = {
        "n_samples": int(len(costs)),
        "expected_loss": float(np.mean(costs)),
        "std_loss": float(np.std(costs)),
        "worst_case_loss": float(np.max(costs)),
        "best_case_loss": float(np.min(costs)),
    }
    for level in (0.95, 0.99):
        var = compute_var(costs, level)
        cvar = compute_cvar(costs, level)
        tag = int(level * 100)
        result[f"VaR_{tag}"] = var
        result[f"CVaR_{tag}"] = cvar
    return result
