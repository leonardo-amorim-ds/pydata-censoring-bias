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

This repository contains only the simulation and selector logic; generated analyses and outputs
are not included.

## Study design

The DGP retains every true outcome before applying the policy. This provides an oracle test
population while reproducing the selected labels available to each practical method.

Three mechanisms drive the experiments:

- **Global drift** changes the general outcome relationship and makes an old model stale.
- **Regional shift** gives a policy-affected subpopulation an outcome relationship that is not
  identifiable from unflagged observations alone.
- **Regional drift** changes that hidden relationship after launch.

Six scenarios also vary trigger rate, holdout percentage and base rate. The reference uses a 1%
base rate, 6% trigger rate, 5% randomized holdout and a strong regional signal. Every method uses
the same LightGBM hyperparameters, and R0–R5 share a six-month rolling window.

## Methods

| ID | Method | Training information |
|---|---|---|
| R0 | Oracle benchmark | Every true outcome, including labels hidden by the policy; unavailable in production |
| R1 | Holdout only | Randomized holdout observations only |
| R2 | No holdout | Unflagged observations only; simulates launching without a randomized holdout |
| R3 | Dropping | Every uncensored observation with equal weight |
| R4 | IPW | The same rows as dropping, with flagged holdout observations weighted by `1 / holdout_pct` |
| R5 | Asymmetric IPW | Trains Dropping and IPW each cycle and selects between them using a rolling window of weighted validation pAUC |
| R6 | No retraining | The model fitted before policy launch |
| R7 | Incremental | The pre-launch model continued with post-launch labels that remain observable |

Here, AIPW means **Asymmetric IPW**, not Augmented IPW. Dropping is the unweighted endpoint
(every observable row has weight 1); pure IPW gives each flagged holdout row weight
`1 / holdout_pct`. Selection never uses test performance.

Asymmetric IPW trains both Dropping and IPW and evaluates them on weighted validation data at every
retraining cycle. This lets the selector use observable non-holdout data instead of relying on the
small holdout alone. The selector averages each method's validation pAUC over the available rolling
window, then deploys the current model from the method with the higher average.

## Code

The runtime code is deliberately split into three small modules:

- `censoring_sim.py` defines the DGP, policy censoring and fitted models.
- `sim_core.py` contains pAUC and training-set construction.
- `study.py` defines the scenario panel, parallel execution, checkpointing and summaries.

## PyData Amsterdam 2026 presentation

Youtube recording: recording link will be added after the conference.
