"""Shared metric and training-set construction for the simulation methods."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


Y_DEFAULT = "y"
MAX_FPR = 0.2
MODES = (
    "benchmark",
    "holdout_only",
    "unflagged_only",
    "drop_unweighted",
    "ipw",
    "pre_launch_only",
)


def partial_auc(y_true, y_scores, max_fpr=MAX_FPR):
    """Area under the ROC curve up to max_fpr, divided by max_fpr."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    fpr_restricted = fpr[:stop]
    tpr_restricted = tpr[:stop]
    if fpr_restricted[-1] < max_fpr:
        fpr_restricted = np.append(fpr_restricted, max_fpr)
        tpr_restricted = np.append(tpr_restricted, np.interp(max_fpr, fpr, tpr))
    return auc(fpr_restricted, tpr_restricted) / max_fpr


def weighted_partial_auc(y_true, y_scores, sample_weight, max_fpr=MAX_FPR):
    """Normalized partial AUC with explicit observation weights."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(
        y_true, y_scores, sample_weight=np.asarray(sample_weight),
        drop_intermediate=True)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    fpr_restricted = fpr[:stop]
    tpr_restricted = tpr[:stop]
    if fpr_restricted[-1] < max_fpr:
        fpr_restricted = np.append(fpr_restricted, max_fpr)
        tpr_restricted = np.append(
            tpr_restricted, np.interp(max_fpr, fpr, tpr))
    return auc(fpr_restricted, tpr_restricted) / max_fpr


def _require_labels(frame, mode):
    if frame[Y_DEFAULT].dtype == object:
        raise ValueError(f"mode={mode}: {Y_DEFAULT} must be bool or numeric")
    missing = int(frame[Y_DEFAULT].isna().sum())
    if missing:
        raise ValueError(f"mode={mode}: {missing} {Y_DEFAULT} labels are NULL")


def build_train_set(
    pool,
    mode,
    window_months,
    anchor_end,
    *,
    features,
    holdout_pct,
    neg_pos_ratio,
    seed,
    launch_date=None,
):
    """Apply the time window, row filter, weights, then negative undersampling."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if anchor_end is None:
        raise ValueError("anchor_end is required")
    if mode == "pre_launch_only" and launch_date is None:
        raise ValueError("pre_launch_only requires launch_date")
    if not 0 < holdout_pct <= 1:
        raise ValueError("holdout_pct must be in (0, 1]")
    _require_labels(pool, mode)

    if window_months is None:
        in_window = np.ones(len(pool), dtype=bool)
    else:
        cutoff = pd.to_datetime(anchor_end) - pd.DateOffset(months=window_months)
        in_window = (pool["og_created"] >= cutoff).values
    flagged = pool["is_flagged"].values
    holdout = pool["is_holdout"].values
    censored = flagged & ~holdout

    if mode == "benchmark":
        keep = in_window
    elif mode == "holdout_only":
        keep = in_window & holdout
    elif mode == "unflagged_only":
        keep = in_window & ~flagged
    elif mode == "pre_launch_only":
        keep = in_window & (pool["og_created"] < pd.to_datetime(launch_date)).values
    else:
        keep = in_window & ~censored

    frame = pool[keep].copy()
    if mode == "ipw":
        frame["sample_weight"] = np.where(
            frame["is_flagged"].values, 1.0 / holdout_pct, 1.0)
    else:
        frame["sample_weight"] = 1.0

    positive = frame[frame[Y_DEFAULT] == 1]
    negative = frame[frame[Y_DEFAULT] == 0]
    if len(positive) < 50:
        raise ValueError(f"mode={mode}: only {len(positive)} positives after filtering")
    n_negative = int(len(positive) * neg_pos_ratio)
    if len(negative) > n_negative:
        negative = negative.sample(n=n_negative, random_state=seed)
    train = pd.concat([positive, negative])
    info = {
        "rows": len(train),
        "positives": len(positive),
    }
    return train[features], train[Y_DEFAULT].values, train.sample_weight.values, info
