# Beyond the Holdout

This repository contains the synthetic study behind **Beyond the Holdout: Mitigating Censoring Bias
with Asymmetric IPW**.

A model-informed policy changes which outcomes can be observed. In fraud, for example, blocked
transactions never produce the counterfactual label needed to evaluate the model or train its
replacement. Metrics computed from the surviving population can therefore look healthy while the
model is getting worse on the population affected by the policy.

The synthetic study asks why a randomized holdout is needed after launch and how its labels should
be used for retraining. The metric is normalized partial AUC through 20% FPR: random is 0.1 and
perfect ranking is 1.0.

## Study design

The DGP retains every true outcome before applying the policy. This provides an oracle test
population while reproducing the selected labels available to each practical method.

Three mechanisms drive the experiments:

- **Global drift** changes the general outcome relationship and makes an old model stale.
- **Regional shift** gives a policy-affected subpopulation an outcome relationship that is not
  identifiable from unflagged observations alone.
- **Regional drift** changes that hidden relationship after launch.

Six scenarios also vary trigger rate, holdout percentage and base rate. The reference uses a 1%
base rate, 6% trigger rate, 5% randomized holdout and a strong regional signal. Every method uses the
same LightGBM configuration and six-month rolling window. Five paired draws are evaluated one and
twelve months after launch, with validation at M and test at M+1.

## Methods

| ID | Method | Training information |
|---|---|---|
| R0 | Oracle benchmark | Every true outcome, including labels hidden by the policy; unavailable in production |
| R1 | Holdout only | Randomized holdout observations only |
| R2 | No holdout | Unflagged observations only; simulates launching without a randomized holdout |
| R3 | Dropping | Every uncensored observation with equal weight |
| R4 | IPW | The same rows as dropping, with flagged holdout observations weighted by `1 / holdout_pct` |
| R5 | Asymmetric IPW | Pooled holdout validation selects dropping or IPW once per scenario-period |
| R6 | No retraining | The model fitted before policy launch |
| R7 | Incremental | The pre-launch model continued with post-launch labels that remain observable |

Here, AIPW means **Asymmetric IPW**, not Augmented IPW. Dropping is the unweighted endpoint
(`alpha = holdout_pct`); pure IPW is the fully corrected endpoint (`alpha = 1`). Selection uses
pooled validation, never test performance or a separate choice inside each replicate.

## Results

Without retraining, drift erodes performance month after month. Retraining alone is insufficient:
metrics from observable labels can disagree materially with randomized-holdout and oracle results.

At the early endpoint, the leading methods statistically tie in all six scenarios. After twelve
months, the information regime determines the winner:

- Dropping wins when censored observations contain no distinct signal. In those three scenarios,
  no-holdout statistically ties dropping; the holdout remains necessary for honest measurement but
  contributes little retraining uplift.
- IPW clearly wins in the two strong-regional-signal scenarios. In the low-base/high-holdout
  scenario it leads numerically in all five draws but remains a two-sided 95% statistical tie.
  Across these IPW-favoring regimes, no-holdout falls behind and weighting recovers part of the
  hidden relationship.

Asymmetric IPW selects the endpoint with the higher mean test score in all six late scenarios.
Across all twelve scenario-period comparisons, it statistically ties the fixed better endpoint and
has the best late average deployable rank.

## Files, modules and notebooks

The completed study used Python 3.11. `requirements.txt` records the package versions printed by the
final successful Databricks execution.

Notebooks:

1. `01_simulation.ipynb` runs or resumes 60 method-comparison tasks and 20 drift diagnostics. Tasks
   are checkpointed before aggregation.
2. `02_results.ipynb` reads the six aggregate CSVs, reproduces the analysis and writes sixteen
   figures.

Final tables are in `sim-results/`; notebook figures are in `sim-figures/`.

The runtime code is deliberately split into three small modules:

- `censoring_sim.py` defines the DGP, policy censoring and fitted models.
- `sim_core.py` contains pAUC and training-set construction.
- `study.py` defines the scenario panel, parallel execution, checkpointing and summaries.

## PyData Amsterdam 2026 presentation

Youtube recording: recording link will be added after the conference.
