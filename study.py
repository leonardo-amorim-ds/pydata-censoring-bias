"""Final scenario panel for the censored-label retraining study."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t

import censoring_sim as sim


PERIODS = {"early": 1, "late": 12}
SEEDS = (0, 1, 2, 3, 4)
STUDY_VERSION = 3
METHOD_ORDER = (
    "R0_benchmark",
    "R1_holdout_only",
    "R2_unflagged_only",
    "R3_drop_unweighted",
    "R4_ipw_pure",
    "R5_asymmetric",
    "R6_no_retrain",
    "R7_incremental",
)
METHOD_LABELS = {
    "R0_benchmark": "uncensored benchmark",
    "R1_holdout_only": "holdout only",
    "R2_unflagged_only": "no holdout",
    "R3_drop_unweighted": "dropping",
    "R4_ipw_pure": "IPW",
    "R5_asymmetric": "Asymmetric IPW",
    "R6_no_retrain": "no retraining",
    "R7_incremental": "incremental",
}

BASE = dict(
    rows_per_month=420_000,
    pre_launch_months=6,
    window_months=6,
    n_valid=420_000,
    n_test=420_000,
    n_features=100,
    n_informative=8,
    risk_strength=1.8,
    region_frac=0.15,
    neg_pos_ratio=30,
)


def _scenario(scenario_id, label, **overrides):
    return {
        "scenario_id": scenario_id,
        "label": label,
        "region_shift": 1.5,
        "drift_per_month": 0.02,
        "region_drift_per_month": 0.0,
        "trigger_rate": 0.06,
        "holdout_pct": 0.05,
        "base_rate": 0.01,
        "policy_noise": 1.8,
        "amount_weight": 0.7,
        **overrides,
    }


SCENARIOS = (
    _scenario("no_shift", "No regional shift", region_shift=0.0),
    _scenario("low_shift", "Low shift + regional drift", region_shift=0.25,
              region_drift_per_month=0.02),
    _scenario("reference", "Reference"),
    _scenario("reference_drift", "Reference + regional drift",
              region_drift_per_month=0.02),
    _scenario("high_trigger_low_holdout", "No shift; 12% trigger; 2% holdout",
              region_shift=0.0, trigger_rate=0.12, holdout_pct=0.02),
    _scenario("low_base_high_holdout", "0.1% base rate; 20% holdout",
              base_rate=0.001, holdout_pct=0.20),
)
DRIFT_LEVELS = (0.0, 0.01, 0.02, 0.04)

_IO_ATTEMPTS = 8


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _delay(attempt):
    time.sleep(min(4.0, 0.25 * 2 ** attempt))


def _read_json(path):
    for attempt in range(_IO_ATTEMPTS):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            if attempt == _IO_ATTEMPTS - 1:
                raise
            _delay(attempt)


def _atomic_json(path, value):
    path = Path(path)
    for attempt in range(_IO_ATTEMPTS):
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{attempt}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(_plain(value), handle, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == _IO_ATTEMPTS - 1:
                raise
            _delay(attempt)


def _atomic_csv(path, frame):
    path = Path(path)
    for attempt in range(_IO_ATTEMPTS):
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{attempt}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == _IO_ATTEMPTS - 1:
                raise
            _delay(attempt)


def scenario_frame():
    return pd.DataFrame(SCENARIOS)


def scenario_config(scenario, period):
    parameters = {key: value for key, value in scenario.items()
                  if key not in ("scenario_id", "label")}
    return {**BASE, **parameters, "months_post_launch": PERIODS[period]}


def validate_design():
    ids = [row["scenario_id"] for row in SCENARIOS]
    if len(ids) != len(set(ids)):
        raise AssertionError("scenario ids must be unique")
    if set(PERIODS) != {"early", "late"} or tuple(PERIODS.values()) != (1, 12):
        raise AssertionError("periods must remain one and twelve months post-launch")
    reference = next(row for row in SCENARIOS if row["scenario_id"] == "reference")
    expected = dict(region_shift=1.5, drift_per_month=0.02, trigger_rate=0.06,
                    holdout_pct=0.05, base_rate=0.01, policy_noise=1.8,
                    amount_weight=0.7)
    if any(reference[key] != value for key, value in expected.items()):
        raise AssertionError("reference scenario changed")
    if not 0 < reference["holdout_pct"] <= 1:
        raise AssertionError("holdout_pct must define a valid IPW endpoint")
    return True


def _task_path(out_dir, scenario_id, period, seed, dgp_seed):
    return Path(out_dir) / "checkpoints" / f"{scenario_id}__{period}__s{seed}__d{dgp_seed}.json"


def _error_path(out_dir, scenario_id, period, seed, dgp_seed):
    return Path(out_dir) / "errors" / f"{scenario_id}__{period}__s{seed}__d{dgp_seed}.json"


def _drift_path(out_dir, drift, seed, dgp_seed):
    code = int(round(1000 * drift))
    return Path(out_dir) / "drift-checkpoints" / f"drift{code:03d}__s{seed}__d{dgp_seed}.json"


def evaluate_task(task, out_dir, threads_per_worker):
    scenario, period, seed, dgp_seed = task
    path = _task_path(out_dir, scenario["scenario_id"], period, seed, dgp_seed)
    if path.exists():
        return {"status": "cached", "path": str(path)}
    try:
        sim.SIM_PARAMS["n_jobs"] = threads_per_worker
        config = scenario_config(scenario, period)
        pool, valid, test, derived = sim.simulate_population(
            config, seed=seed, dgp_seed=dgp_seed)
        pool = sim.apply_policy(pool, derived, seed)
        valid = sim.apply_policy(valid, derived, seed)
        methods = sim.run_methods(pool, valid, test, derived, seed=seed)
        _atomic_json(path, {
            "scenario_id": scenario["scenario_id"],
            "period": period,
            "seed": seed,
            "dgp_seed": dgp_seed,
            "methods": methods.to_dict("records"),
        })
        _error_path(out_dir, scenario["scenario_id"], period, seed, dgp_seed).unlink(
            missing_ok=True)
        return {"status": "ok", "path": str(path)}
    except Exception as exc:
        _atomic_json(_error_path(out_dir, scenario["scenario_id"], period, seed, dgp_seed), {
            "task": task,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        return {"status": "error", "error": repr(exc)}


def evaluate_drift_task(task, out_dir, threads_per_worker):
    drift, seed, dgp_seed = task
    path = _drift_path(out_dir, drift, seed, dgp_seed)
    if path.exists():
        return {"status": "cached", "path": str(path)}
    try:
        sim.SIM_PARAMS["n_jobs"] = threads_per_worker
        scenario = next(row for row in SCENARIOS if row["scenario_id"] == "no_shift")
        config = scenario_config(scenario, "late")
        config["drift_per_month"] = drift
        pool, valid, _, derived = sim.simulate_population(
            config, seed=seed, dgp_seed=dgp_seed)
        pool = sim.apply_policy(pool, derived, seed)
        trajectory = sim.no_retrain_trajectory(pool, valid, derived, seed)
        _atomic_json(path, {
            "drift_per_month": drift,
            "seed": seed,
            "dgp_seed": dgp_seed,
            "trajectory": trajectory.to_dict("records"),
        })
        code = int(round(1000 * drift))
        (Path(out_dir) / "drift-errors" /
         f"drift{code:03d}__s{seed}__d{dgp_seed}.json").unlink(missing_ok=True)
        return {"status": "ok", "path": str(path)}
    except Exception as exc:
        code = int(round(1000 * drift))
        error_path = Path(out_dir) / "drift-errors" / f"drift{code:03d}__s{seed}__d{dgp_seed}.json"
        _atomic_json(error_path, {
            "task": task, "error": repr(exc), "traceback": traceback.format_exc(),
        })
        return {"status": "error", "error": repr(exc)}


def load_results(out_dir):
    return [_read_json(path) for path in sorted(
        glob.glob(str(Path(out_dir) / "checkpoints" / "*.json")))]


def _raw_frames(records):
    method_rows = []
    for record in records:
        keys = {key: record[key] for key in ("scenario_id", "period", "seed", "dgp_seed")}
        method_rows.extend([keys | row for row in record["methods"]])
    raw = pd.DataFrame(method_rows)
    if raw.empty:
        return raw

    asymmetric = []
    keys = ["scenario_id", "period", "dgp_seed"]
    for _, group in raw[raw.method.isin(
            ["R3_drop_unweighted", "R4_ipw_pure"])].groupby(keys):
        selected = group.groupby("method").valid_pAUC.mean().idxmax()
        chosen = group[group.method == selected].copy()
        chosen["method"] = "R5_asymmetric"
        chosen["selected_endpoint"] = selected
        asymmetric.append(chosen)
    raw["selected_endpoint"] = None
    return pd.concat([raw, *asymmetric], ignore_index=True)


def _bounds(values):
    values = pd.Series(values).dropna()
    if len(values) < 2:
        return np.nan, np.nan
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean() - half, values.mean() + half


def _status(values):
    lower, upper = _bounds(values)
    if lower > 0:
        return "wins"
    if upper < 0:
        return "loses"
    return "ties"


def summarize_methods(raw):
    keys = ["scenario_id", "period", "dgp_seed", "method"]
    summary = (raw.groupby(keys, as_index=False)
               .agg(test_mean=("test_pAUC", "mean"), test_sd=("test_pAUC", "std"),
                    valid_mean=("valid_pAUC", "mean"), n=("seed", "nunique"),
                    training_rows=("rows", "mean"), positives=("positives", "mean"),
                    selected_endpoint=("selected_endpoint", "first")))
    summary = summary.merge(scenario_frame(), on="scenario_id", how="left")
    ceiling = (summary[summary.method == "R0_benchmark"]
               [["scenario_id", "period", "dgp_seed", "test_mean"]]
               .rename(columns={"test_mean": "ceiling"}))
    summary = summary.merge(ceiling, on=["scenario_id", "period", "dgp_seed"])
    summary["pct_of_ceiling"] = summary.test_mean / summary.ceiling
    summary["method_label"] = summary.method.map(METHOD_LABELS)
    deployable = summary.method != "R0_benchmark"
    summary.loc[deployable, "rank"] = summary[deployable].groupby(
        ["scenario_id", "period", "dgp_seed"])["test_mean"].rank(
            ascending=False, method="min")
    best = summary[deployable].groupby(
        ["scenario_id", "period", "dgp_seed"])["test_mean"].transform("max")
    summary.loc[deployable, "regret_vs_best_deployable"] = (
        summary.loc[deployable, "test_mean"] - best)
    return summary


def summarize_endpoints(raw):
    rows = []
    for keys, group in raw.groupby(["scenario_id", "period", "dgp_seed"]):
        wide = group.pivot(index="seed", columns="method", values="test_pAUC")
        delta = wide.R4_ipw_pure - wide.R3_drop_unweighted
        lower, upper = _bounds(delta)
        rows.append({
            "scenario_id": keys[0], "period": keys[1], "dgp_seed": keys[2],
            "n": len(delta), "dropping_mean": wide.R3_drop_unweighted.mean(),
            "ipw_mean": wide.R4_ipw_pure.mean(), "ipw_minus_dropping": delta.mean(),
            "paired_sd": delta.std(ddof=1), "lower_95": lower, "upper_95": upper,
            "negative_seeds": int((delta < 0).sum()),
            "positive_seeds": int((delta > 0).sum()),
            "result": ("ipw_wins" if lower > 0 else
                       "dropping_wins" if upper < 0 else "tie"),
        })
    return scenario_frame().merge(pd.DataFrame(rows), on="scenario_id", how="right")


def summarize_asymmetric(raw, method_summary):
    rows = []
    for keys, group in raw.groupby(["scenario_id", "period", "dgp_seed"]):
        wide = group.pivot(index="seed", columns="method", values="test_pAUC")
        endpoint = max(
            ("R3_drop_unweighted", wide.R3_drop_unweighted.mean()),
            ("R4_ipw_pure", wide.R4_ipw_pure.mean()), key=lambda item: item[1])[0]
        gap = wide.R5_asymmetric - wide[endpoint]
        lower, upper = _bounds(gap)
        selected = group.loc[group.method == "R5_asymmetric", "selected_endpoint"].dropna().iloc[0]
        cell = method_summary[(method_summary.scenario_id == keys[0])
                              & (method_summary.period == keys[1])
                              & (method_summary.dgp_seed == keys[2])]
        alternatives = cell[~cell.method.isin(["R0_benchmark", "R5_asymmetric"])]
        best_other = alternatives.loc[alternatives.test_mean.idxmax()]
        non_endpoints = cell[cell.method.isin(
            ["R1_holdout_only", "R2_unflagged_only", "R6_no_retrain", "R7_incremental"])]
        best_non_endpoint = non_endpoints.loc[non_endpoints.test_mean.idxmax()]
        asymmetric_mean = cell.loc[cell.method == "R5_asymmetric", "test_mean"].iloc[0]
        non_endpoint_gap = wide.R5_asymmetric - wide[best_non_endpoint.method]
        non_endpoint_lower, non_endpoint_upper = _bounds(non_endpoint_gap)
        rows.append({
            "scenario_id": keys[0], "period": keys[1], "dgp_seed": keys[2],
            "n": len(gap), "selected_endpoint": selected, "better_endpoint": endpoint,
            "gap_vs_better_endpoint": gap.mean(), "paired_sd": gap.std(ddof=1),
            "lower_95": lower, "upper_95": upper, "result_vs_endpoint": _status(gap),
            "best_other_method": best_other.method,
            "gap_vs_best_other": asymmetric_mean - best_other.test_mean,
            "best_non_endpoint_method": best_non_endpoint.method,
            "gap_vs_best_non_endpoint": asymmetric_mean - best_non_endpoint.test_mean,
            "non_endpoint_lower_95": non_endpoint_lower,
            "non_endpoint_upper_95": non_endpoint_upper,
            "result_vs_non_endpoint": _status(non_endpoint_gap),
            "asymmetric_rank": int(cell.loc[cell.method == "R5_asymmetric", "rank"].iloc[0]),
        })
    return scenario_frame().merge(pd.DataFrame(rows), on="scenario_id", how="right")


def save_outputs(out_dir):
    records = load_results(out_dir)
    raw = _raw_frames(records)
    if raw.empty:
        return {}
    methods = summarize_methods(raw)
    endpoints = summarize_endpoints(raw)
    asymmetric = summarize_asymmetric(raw, methods)
    outputs = {
        "method_raw": raw,
        "method_summary": methods,
        "endpoint_summary": endpoints,
        "asymmetric_summary": asymmetric,
    }
    for name, frame in outputs.items():
        _atomic_csv(Path(out_dir) / f"{name}.csv", frame)
    return outputs


def run_drift_study(here, out_name="sim-results", n_workers=5,
                    threads_per_worker=12, seeds=SEEDS, dgp_seed=0):
    out_dir = Path(here) / out_name
    tasks = [(drift, seed, dgp_seed) for drift in DRIFT_LEVELS for seed in seeds]
    pending = [task for task in tasks if not _drift_path(out_dir, *task).exists()]
    progress_path = out_dir / "drift_progress.json"
    prior_progress = _read_json(progress_path) if progress_path.exists() else {}
    started = time.time()
    for offset in range(0, len(pending), n_workers):
        batch = pending[offset:offset + n_workers]
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=10)(
            delayed(evaluate_drift_task)(task, out_dir, threads_per_worker) for task in batch)
        _atomic_json(progress_path, {
            "stage": "running", "completed": len(tasks) - len(pending) + offset + len(batch),
            "total": len(tasks), "batch_errors": sum(
                row["status"] == "error" for row in results),
            "elapsed_minutes": (time.time() - started) / 60,
        })

    records = [_read_json(_drift_path(out_dir, *task)) for task in tasks
               if _drift_path(out_dir, *task).exists()]
    rows = []
    for record in records:
        keys = {key: record[key] for key in ("drift_per_month", "seed", "dgp_seed")}
        rows.extend([keys | point for point in record["trajectory"]])
    trajectory = pd.DataFrame(rows)
    if len(trajectory):
        _atomic_csv(out_dir / "drift_trajectory.csv", trajectory)
    errors = len(glob.glob(str(out_dir / "drift-errors" / "*.json")))
    elapsed = ((time.time() - started) / 60 if pending
               else prior_progress.get("elapsed_minutes", 0.0))
    _atomic_json(progress_path, {
        "stage": "complete" if not errors else "complete_with_errors",
        "completed": len(records), "total": len(tasks), "errors": errors,
        "elapsed_minutes": elapsed,
    })
    return trajectory


def _source_hash(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def run_study(here, out_name="sim-results", n_workers=5, threads_per_worker=12,
              seeds=SEEDS, dgp_seed=0):
    validate_design()
    here = Path(here)
    out_dir = here / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(out_dir / "scenario_definitions.csv", scenario_frame())
    manifest = {
        "study_version": STUDY_VERSION,
        "interval": "two-sided 95%",
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "seeds": list(seeds),
        "dgp_seed": dgp_seed,
        "source_hashes": {name: _source_hash(here / name)
                          for name in ("study.py", "censoring_sim.py", "sim_core.py")},
    }
    manifest_path = out_dir / "manifest.json"
    existing = glob.glob(str(out_dir / "checkpoints" / "*.json"))
    if manifest_path.exists() and existing:
        saved = _read_json(manifest_path)
        comparable = ("study_version", "seeds", "dgp_seed", "source_hashes")
        changed = [key for key in comparable if saved.get(key) != manifest.get(key)]
        if changed:
            raise RuntimeError(
                f"existing checkpoints are incompatible ({', '.join(changed)}); "
                "use a clean output directory")
    else:
        _atomic_json(manifest_path, manifest)

    tasks = [(scenario, period, seed, dgp_seed)
             for scenario in SCENARIOS for period in PERIODS for seed in seeds]
    pending = [task for task in tasks if not _task_path(
        out_dir, task[0]["scenario_id"], task[1], task[2], task[3]).exists()]
    progress_path = out_dir / "progress.json"
    prior_progress = _read_json(progress_path) if progress_path.exists() else {}
    started = time.time()
    for offset in range(0, len(pending), n_workers):
        batch = pending[offset:offset + n_workers]
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=10)(
            delayed(evaluate_task)(task, out_dir, threads_per_worker) for task in batch)
        completed = len(tasks) - len(pending) + offset + len(batch)
        _atomic_json(progress_path, {
            "stage": "running", "completed": completed, "total": len(tasks),
            "errors": len(glob.glob(str(out_dir / "errors" / "*.json"))),
            "elapsed_minutes": (time.time() - started) / 60,
            "batch_errors": sum(row["status"] == "error" for row in results),
        })
        if (offset // n_workers + 1) % 5 == 0:
            save_outputs(out_dir)

    outputs = save_outputs(out_dir)
    errors = len(glob.glob(str(out_dir / "errors" / "*.json")))
    elapsed = ((time.time() - started) / 60 if pending
               else prior_progress.get("elapsed_minutes", 0.0))
    _atomic_json(progress_path, {
        "stage": "complete" if not errors else "complete_with_errors",
        "completed": len(glob.glob(str(out_dir / "checkpoints" / "*.json"))),
        "total": len(tasks), "errors": errors,
        "elapsed_minutes": elapsed,
    })
    return {"out_dir": str(out_dir), **outputs}
