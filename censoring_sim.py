"""Static data-generating process and fitted methods for the final study."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

import sim_core as core


Y = core.Y_DEFAULT
ORIGIN = pd.Timestamp("2025-01-01")

SIM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "n_jobs": -1,
}

CONFIG_KEYS = frozenset({
    "rows_per_month", "pre_launch_months", "months_post_launch", "window_months",
    "n_valid", "n_test", "n_features", "n_informative", "base_rate",
    "risk_strength", "policy_noise", "amount_weight", "trigger_rate", "holdout_pct",
    "neg_pos_ratio", "region_shift", "region_frac", "drift_per_month",
    "region_drift_per_month",
})

STANDARD_METHODS = (
    ("R0_benchmark", "benchmark"),
    ("R1_holdout_only", "holdout_only"),
    ("R2_unflagged_only", "unflagged_only"),
    ("R3_drop_unweighted", "drop_unweighted"),
)


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _rotate(base, toward, per_month, last_month):
    theta = np.clip(per_month * np.arange(last_month + 1), 0, 1) * np.pi / 2
    directions = np.cos(theta)[:, None] * base + np.sin(theta)[:, None] * toward
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _risk_directions(dgp, dimensions, informative, last_month, drift_per_month):
    base = np.zeros(dimensions)
    base[:informative] = dgp.normal(size=informative)
    base = _unit(base)
    toward = np.zeros(dimensions)
    toward[:informative] = dgp.normal(size=informative)
    toward = _unit(toward - toward.dot(base) * base)
    return base, _rotate(base, toward, drift_per_month, last_month)


def _region_directions(dgp, dimensions, global_base, last_month, drift_per_month):
    base = dgp.normal(size=dimensions)
    base = _unit(base - base.dot(global_base) * global_base)
    toward = dgp.normal(size=dimensions)
    toward = toward - toward.dot(global_base) * global_base
    toward = _unit(toward - toward.dot(base) * base)
    return _rotate(base, toward, drift_per_month, last_month)


def _standardize_monthly(values, months):
    standardized = np.empty_like(values, dtype=float)
    for month in np.unique(months):
        rows = months == month
        standardized[rows] = (values[rows] - values[rows].mean()) / values[rows].std()
    return standardized


def _intercept(values, base_rate, rng):
    sample = values if len(values) <= 30_000 else values[
        rng.choice(len(values), 30_000, replace=False)]
    low, high = -30.0, 30.0
    for _ in range(50):
        middle = (low + high) / 2
        probability = 1 / (1 + np.exp(-(middle + sample)))
        if probability.mean() > base_rate:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def simulate_population(config, seed=0, dgp_seed=0):
    """Return pool, validation, test and derived configuration for one replicate."""
    config = dict(config)
    missing = CONFIG_KEYS - config.keys()
    extra = config.keys() - CONFIG_KEYS
    if missing or extra:
        raise ValueError(f"invalid simulation config; missing={sorted(missing)}, extra={sorted(extra)}")
    rng = np.random.default_rng(seed)
    dgp = np.random.default_rng(dgp_seed)
    n_valid = int(config["n_valid"])
    n_test = int(config["n_test"])
    n_months = int(config["pre_launch_months"] + config["months_post_launch"])
    n_pool = int(config["rows_per_month"] * n_months)
    n = n_pool + n_valid + n_test
    dimensions = int(config["n_features"])

    features = rng.normal(size=(n, dimensions))
    global_base, global_directions = _risk_directions(
        dgp, dimensions, int(config["n_informative"]), n_months + 1,
        config["drift_per_month"])

    permutation = rng.permutation(n)
    month = np.zeros(n, dtype=int)
    month[permutation[:n_pool]] = rng.integers(0, n_months, n_pool)
    month[permutation[n_pool:n_pool + n_valid]] = n_months
    month[permutation[n_pool + n_valid:]] = n_months + 1

    risk = np.einsum("ij,ij->i", features, global_directions[month])
    amount = rng.lognormal(mean=2.0, sigma=0.8, size=n)
    if config["region_shift"] > 0:
        regional_directions = _region_directions(
            dgp, dimensions, global_base, n_months + 1,
            config["region_drift_per_month"])
        threshold = float(np.quantile(amount, 1 - config["region_frac"]))
        in_region = amount > threshold
        regional_term = in_region * np.einsum(
            "ij,ij->i", features, regional_directions[month])
        risk = risk + config["region_shift"] * regional_term

    scaled_risk = config["risk_strength"] * risk
    intercepts = np.array([
        _intercept(scaled_risk[month == value], config["base_rate"], rng)
        for value in range(n_months + 2)
    ])
    probability = 1 / (1 + np.exp(-(intercepts[month] + scaled_risk)))
    outcome = rng.random(n) < probability

    noisy_risk = risk + rng.normal(scale=config["policy_noise"], size=n)
    risk_score = _standardize_monthly(noisy_risk, month)
    amount_score = _standardize_monthly(np.log(amount), month)
    policy_score = ((1 - config["amount_weight"]) * risk_score
                    + config["amount_weight"] * amount_score)

    frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(dimensions)])
    frame["log_amount"] = np.log(amount)
    frame[Y] = outcome
    frame["policy_score"] = policy_score
    frame["month"] = month
    frame["og_created"] = ORIGIN + pd.to_timedelta(month * 30, unit="D")

    pool = frame.iloc[permutation[:n_pool]].reset_index(drop=True)
    valid = frame.iloc[permutation[n_pool:n_pool + n_valid]].reset_index(drop=True)
    test = (frame.iloc[permutation[n_pool + n_valid:]].reset_index(drop=True)
            .drop(columns="policy_score"))
    config.update({
        "launch_month": int(config["pre_launch_months"]),
        "launch_date": ORIGIN + pd.to_timedelta(config["pre_launch_months"] * 30, unit="D"),
        "anchor_end": ORIGIN + pd.to_timedelta(n_months * 30, unit="D"),
    })
    return pool, valid, test, config


def apply_policy(frame, config, seed=0):
    """Apply the monthly policy threshold and independent randomized holdout."""
    rng = np.random.default_rng(seed + 977)
    frame = frame.copy()
    live = frame.month.values >= config["launch_month"]
    flagged = np.zeros(len(frame), dtype=bool)
    for month in np.unique(frame.loc[live, "month"]):
        rows = live & (frame.month.values == month)
        threshold = np.quantile(frame.loc[rows, "policy_score"], 1 - config["trigger_rate"])
        flagged[rows] = frame.loc[rows, "policy_score"].values > threshold
    frame["is_flagged"] = flagged
    frame["is_holdout"] = rng.random(len(frame)) < config["holdout_pct"]
    return frame.drop(columns="policy_score")


def _features(pool):
    columns = [column for column in pool if column.startswith("f")]
    return columns + ["log_amount"]


def _scores(model, valid, test, features, holdout_pct, benchmark=False):
    valid_prediction = model.predict_proba(valid[features])[:, 1]
    holdout = valid.is_holdout.values
    unflagged = ~valid.is_flagged.values
    observable = unflagged | holdout
    observable_frame = valid.loc[observable]
    observation_weight = np.where(
        observable_frame.is_flagged.values, 1.0 / holdout_pct, 1.0)
    holdout_score = core.partial_auc(valid.loc[holdout, Y], valid_prediction[holdout])
    observed_score = core.partial_auc(valid.loc[unflagged, Y], valid_prediction[unflagged])
    oracle_score = core.partial_auc(valid[Y], valid_prediction)
    return {
        "valid_pAUC": oracle_score if benchmark else holdout_score,
        "holdout_valid_pAUC": holdout_score,
        "observed_valid_pAUC": observed_score,
        "weighted_valid_pAUC": core.weighted_partial_auc(
            observable_frame[Y], valid_prediction[observable], observation_weight),
        "oracle_valid_pAUC": oracle_score,
        "test_pAUC": (core.partial_auc(
            test[Y], model.predict_proba(test[features])[:, 1])
            if test is not None else np.nan),
    }


def _fit(pool, valid, test, features, mode, config, seed, full_history=False):
    X, y, weights, info = core.build_train_set(
        pool,
        mode,
        None if full_history else config["window_months"],
        config["anchor_end"],
        features=features,
        holdout_pct=config["holdout_pct"],
        neg_pos_ratio=config["neg_pos_ratio"],
        seed=seed,
        launch_date=config["launch_date"],
    )
    model = LGBMClassifier(**SIM_PARAMS, random_state=seed)
    model.fit(X, y, sample_weight=weights)
    return {
        **_scores(model, valid, test, features, config["holdout_pct"],
                  benchmark=mode == "benchmark"),
        **info,
    }


def _fit_incremental(pool, valid, test, features, config, seed):
    pre_launch = pool[pool.og_created < config["launch_date"]]
    post_launch = pool[pool.og_created >= config["launch_date"]]
    X0, y0, w0, _ = core.build_train_set(
        pre_launch, "benchmark", None, config["anchor_end"], features=features,
        holdout_pct=config["holdout_pct"], neg_pos_ratio=config["neg_pos_ratio"],
        seed=seed)
    base = LGBMClassifier(**SIM_PARAMS, random_state=seed).fit(X0, y0, sample_weight=w0)
    X1, y1, w1, info = core.build_train_set(
        post_launch, "drop_unweighted", config["window_months"], config["anchor_end"],
        features=features,
        holdout_pct=config["holdout_pct"], neg_pos_ratio=config["neg_pos_ratio"],
        seed=seed)
    continued = LGBMClassifier(**SIM_PARAMS, random_state=seed)
    valid_holdout = valid[valid.is_holdout]
    continued.fit(
        X1, y1, sample_weight=w1, init_model=base.booster_,
        eval_set=[(valid_holdout[features], valid_holdout[Y])], eval_metric="auc",
        callbacks=[early_stopping(20, verbose=False)])
    return {
        **_scores(continued, valid, test, features, config["holdout_pct"]),
        **info,
    }


def run_methods(pool, valid, test, config, seed=0):
    """Fit the seven actual models. Asymmetric IPW is selected during aggregation."""
    features = _features(pool)
    rows = []

    def record(method, values):
        rows.append({
            "method": method,
            "seed": seed,
            **values,
        })

    for method, mode in STANDARD_METHODS:
        record(method, _fit(pool, valid, test, features, mode, config, seed))
    record("R4_ipw_pure", _fit(
        pool, valid, test, features, "ipw", config, seed))
    record("R6_no_retrain", _fit(
        pool, valid, test, features, "pre_launch_only", config, seed,
        full_history=True))
    record("R7_incremental", _fit_incremental(
        pool, valid, test, features, config, seed))
    return pd.DataFrame(rows)


def run_selector_endpoints(pool, valid, config, seed=0):
    """Fit the two Asymmetric-IPW candidates for one realized cycle."""
    features = _features(pool)
    rows = []
    for method, mode in (
        ("R3_drop_unweighted", "drop_unweighted"),
        ("R4_ipw_pure", "ipw"),
    ):
        rows.append({
            "method": method,
            "seed": seed,
            **_fit(pool, valid, None, features, mode, config, seed),
        })
    return pd.DataFrame(rows)


def no_retrain_trajectory(pool, valid, config, seed=0):
    """Fit once on pre-launch data and score months 0–12 after launch."""
    features = _features(pool)
    pre_launch = pool[pool.og_created < config["launch_date"]]
    X, y, weights, _ = core.build_train_set(
        pre_launch, "benchmark", None, config["anchor_end"], features=features,
        holdout_pct=config["holdout_pct"], neg_pos_ratio=config["neg_pos_ratio"],
        seed=seed)
    model = LGBMClassifier(**SIM_PARAMS, random_state=seed)
    model.fit(X, y, sample_weight=weights)

    evaluation = pd.concat([
        pool.loc[pool.month >= config["launch_month"], features + [Y, "month"]],
        valid[features + [Y, "month"]],
    ], ignore_index=True)
    rows = []
    for month, frame in evaluation.groupby("month"):
        months_since_launch = int(month - config["launch_month"])
        if 0 <= months_since_launch <= 12:
            rows.append({
                "month": months_since_launch,
                "pAUC": core.partial_auc(
                    frame[Y], model.predict_proba(frame[features])[:, 1]),
                "rows": len(frame),
                "positives": int(frame[Y].sum()),
            })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
