# State-or-Space: final public results

## Protocol

- Tasks: 30-day readmission and ICU transfer.
- Frozen task-specific models: RETAIN-lite numeric for readmission and two-layer numeric GRU for ICU transfer.
- Seeds: 42, 43, 44, 45 and 46.
- Prediction boundary: `event_time <= prediction_time`.
- Platt calibration fitted on tuning only; final metrics computed on held-out only.
- Patient-cluster bootstrap: 10,000 repetitions.

## Main findings

1. The complete condition-era representation did not show a statistically established overall advantage over raw input at context 4096. Significant metrics for the full comparison were: readmission — none; ICU transfer — none.
2. For ICU transfer, explicit state features were significantly better than the structure-null control for: AUROC, AUPRC, Brier score, LogLoss, Top-10% precision.
3. For ICU transfer, backfill was significantly better than no-backfill for: AUROC, LogLoss.
4. Copy-forward robustness was task-dependent. At 100% copy-forward, mean absolute probability change was 0.001216 for raw and 0.000304 for condition era in readmission. For ICU transfer, the corresponding values were 0.001904 and 0.002724.
5. All 9/9 split-audit checks passed, and all 132/132 representation invariants passed.

## Publication scope

This directory contains aggregate metrics, confidence intervals, figures and protocol summaries only. It contains no patient-level or episode-level predictions, identifiers, diagnosis codes, checkpoints or raw EHR data.
