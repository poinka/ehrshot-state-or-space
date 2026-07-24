# Public results

This directory is a publication-safe export of the State-or-Space experiment.

Included:

- aggregate ensemble metrics;
- patient-bootstrap comparison summaries;
- aggregate copy-forward robustness summaries;
- two publication figures in PNG and PDF;
- protocol and integrity summaries;
- SHA-256 hashes.

Not included:

- patient-level or episode-level predictions;
- `row_id`, `subject_id`, prediction timestamps or diagnosis codes;
- EHRSHOT/MEDS rows or sequence examples;
- checkpoints;
- copy-forward episode plans or eligible-visit lists;
- private ClearML task identifiers or internal storage manifests.

The private reproducibility package remains in restricted MinIO/ClearML storage.
The code and frozen configuration files remain in the repository itself.
