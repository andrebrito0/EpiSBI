from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _logit(x, lo, hi, eps=1e-6):
    p = np.clip((x - lo) / (hi - lo), eps, 1 - eps)
    return np.log(p / (1 - p))


def _inv_logit(y, lo, hi):
    p = 1 / (1 + np.exp(-y))
    return lo + p * (hi - lo)


def regression_adjust(
    thetas: np.ndarray,
    summaries: np.ndarray,
    obs_summary: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    param_names: Optional[List[str]] = None,
    method: str = "loclinear",
    ridge_alphas: Optional[np.ndarray] = None,
):
    # Post-hoc regression adjustment for an already-accepted ABC sample
    # (Beaumont, Zhang & Balding 2002): fits a locally weighted regression of
    # each parameter on the summaries within the accepted set, in logit space
    # to respect the prior bounds, then shifts each draw to where its
    # summaries would exactly equal the observed data. Reuses simulations
    # already run by run_abc -- no extra simulation cost, and typically
    # de-biases and sharpens the posterior, particularly at a loose
    # acceptance tolerance.
    #
    # thetas, summaries: the accepted ABC sample (n_accepted, n_params) and
    #   its summaries (n_accepted, n_summaries) -- e.g. run_abc's
    #   refined_thetas and the summaries used to accept them.
    # bounds: prior bounds per parameter, e.g. {name: (low, high), ...} --
    #   can be taken directly from a UniformPrior's .bounds attribute.
    # method: "loclinear" (Beaumont et al. 2002) or "ridge" (regularised,
    #   more robust when summaries are correlated).
    #
    # Requires scikit-learn (a transitive dependency via sbi); imported
    # lazily so importing this module never requires it unless this
    # function is actually called.
    from sklearn.linear_model import LinearRegression, RidgeCV

    thetas = np.asarray(thetas, dtype=float)
    summaries = np.asarray(summaries, dtype=float)
    obs_summary = np.asarray(obs_summary, dtype=float)

    if param_names is None:
        param_names = list(bounds.keys())
    if method not in ("loclinear", "ridge"):
        raise ValueError(f"method must be 'loclinear' or 'ridge', got {method!r}.")

    sds = summaries.std(axis=0)
    sds[sds == 0] = 1
    diff = (summaries / sds) - (obs_summary / sds)

    # Epanechnikov kernel weights from standardised distance to the observed
    # summaries -- points closer to the observed data count more in the fit.
    dist = np.sqrt((diff**2).sum(axis=1))
    d_max = dist.max()
    weights = np.clip(1 - (dist / d_max) ** 2, 0, None) if d_max > 0 else np.ones_like(dist)

    adjusted = np.empty_like(thetas)
    for j, name in enumerate(param_names):
        lo, hi = bounds[name]
        y = _logit(thetas[:, j], lo, hi)
        if method == "ridge":
            alphas = np.logspace(-3, 3, 13) if ridge_alphas is None else ridge_alphas
            model = RidgeCV(alphas=alphas)
        else:
            model = LinearRegression()
        model.fit(diff, y, sample_weight=weights)
        adjusted_y = y - diff @ np.asarray(model.coef_).reshape(-1)
        adjusted[:, j] = _inv_logit(adjusted_y, lo, hi)

    return adjusted
