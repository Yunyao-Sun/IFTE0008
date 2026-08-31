"""Elastic Net feature selection utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from config import CFG


def elastic_net_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    在训练集上 fit EN，返回选中的特征索引和名称。
    只应在训练集调用一次，验证/测试集用 elastic_net_apply 复用结果。

    返回：
      selected_idx   : 选中列的索引数组 (np.ndarray)
      selected_names : 对应的特征名列表
    """
    top_n = top_n or CFG.EN_TOP_N

    if X.shape[1] <= top_n:
        idx = np.arange(X.shape[1])
        return idx, list(feature_names)

    X_scaled = StandardScaler().fit_transform(X)
    model = ElasticNet(
        alpha       = CFG.EN_ALPHA,
        l1_ratio    = CFG.EN_L1_RATIO,
        max_iter    = 5000,
        random_state = CFG.SEED,
    )
    model.fit(X_scaled, y)

    coef_abs = np.abs(model.coef_)
    if np.all(coef_abs == 0):
        selected_idx = np.arange(top_n)
    else:
        selected_idx = np.argsort(coef_abs)[-top_n:]
        selected_idx = selected_idx[np.argsort(coef_abs[selected_idx])[::-1]]

    selected_names = [feature_names[i] for i in selected_idx]
    return selected_idx, selected_names


def elastic_net_apply(
    X: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    """
    用 elastic_net_fit 返回的索引筛选特征。
    可用于验证集和测试集，无需重新 fit。

    X: (T, n_features)
    """
    return X[:, selected_idx]


def elastic_net_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    兼容接口：fit + apply 合并（保留原始接口，dataset_builder 旧版调用）。
    注意：dataset_builder 修复版已改为 fit/apply 分离，此函数仅作备用。
    """
    idx, names = elastic_net_fit(X, y, feature_names, top_n)
    return elastic_net_apply(X, idx), names
