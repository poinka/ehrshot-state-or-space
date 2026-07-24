| task_label | representation_label | model | max_len | era_gap | n_patients | n_positive | auroc | auprc | brier | logloss | top_10pct_precision | top_10pct_lift | top_10pct_event_capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICU transfer | Condition era 180 + backfill, context 4096 | GRU_2L_numeric | 4096 | 180.000000 | 1154 | 85 | 0.775699 | 0.236227 | 0.036339 | 0.149947 | 0.196078 | 4.698962 | 0.470588 |
| ICU transfer | Condition era 30 + backfill, context 4096 | GRU_2L_numeric | 4096 | 30.000000 | 1154 | 85 | 0.755027 | 0.160087 | 0.037649 | 0.155652 | 0.156863 | 3.759170 | 0.376471 |
| ICU transfer | Condition era 90 + backfill, context 4096 | GRU_2L_numeric | 4096 | 90.000000 | 1154 | 85 | 0.801302 | 0.224124 | 0.036185 | 0.147454 | 0.181373 | 4.346540 | 0.435294 |
| ICU transfer | Condition era 90 without backfill, context 4096 | GRU_2L_numeric | 4096 | 90.000000 | 1154 | 85 | 0.773162 | 0.200678 | 0.036819 | 0.151755 | 0.166667 | 3.994118 | 0.400000 |
| ICU transfer | Raw, context 4096 | GRU_2L_numeric | 4096 | nan | 1154 | 85 | 0.794829 | 0.219400 | 0.036366 | 0.149259 | 0.176471 | 4.229066 | 0.423529 |
| ICU transfer | Structure-null, context 4096 | GRU_2L_numeric | 4096 | 90.000000 | 1154 | 85 | 0.770419 | 0.182534 | 0.037309 | 0.155279 | 0.156863 | 3.759170 | 0.376471 |
| ICU transfer | Condition era 90 + backfill, context 16384 | GRU_2L_numeric | 16384 | 90.000000 | 1154 | 85 | 0.777435 | 0.235241 | 0.036149 | 0.150533 | 0.176471 | 4.229066 | 0.423529 |
| ICU transfer | Raw, context 16384 | GRU_2L_numeric | 16384 | nan | 1154 | 85 | 0.780509 | 0.201703 | 0.036903 | 0.153049 | 0.181373 | 4.346540 | 0.435294 |
| Readmission | Condition era 90 + backfill, context 4096 | RETAIN_lite_numeric | 4096 | 90.000000 | 1190 | 260 | 0.798487 | 0.420572 | 0.086019 | 0.296925 | 0.465753 | 3.921286 | 0.392308 |
| Readmission | Condition era 90 without backfill, context 4096 | RETAIN_lite_numeric | 4096 | 90.000000 | 1190 | 260 | 0.800738 | 0.405151 | 0.087104 | 0.299147 | 0.470320 | 3.959730 | 0.396154 |
| Readmission | Raw, context 4096 | RETAIN_lite_numeric | 4096 | nan | 1190 | 260 | 0.794152 | 0.409727 | 0.086902 | 0.299557 | 0.452055 | 3.805954 | 0.380769 |
| Readmission | Structure-null, context 4096 | RETAIN_lite_numeric | 4096 | 90.000000 | 1190 | 260 | 0.792810 | 0.412601 | 0.086882 | 0.299961 | 0.456621 | 3.844398 | 0.384615 |
| Readmission | Condition era 90 + backfill, context 16384 | RETAIN_lite_numeric | 16384 | 90.000000 | 1190 | 260 | 0.794186 | 0.416137 | 0.087667 | 0.303417 | 0.465753 | 3.921286 | 0.392308 |
| Readmission | Raw, context 16384 | RETAIN_lite_numeric | 16384 | nan | 1190 | 260 | 0.803421 | 0.420164 | 0.088041 | 0.305622 | 0.438356 | 3.690622 | 0.369231 |