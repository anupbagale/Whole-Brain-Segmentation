# Test results

Split: 100 train, 18 validation, 30 test scans (seed 0).

Mean Dice per subject, averaged over structures, then mean ± SD over test subjects.

| Model | All its structures | 27 shared structures |
|---|---|---|
| 45 classes | 0.733 ± 0.057 | 0.797 ± 0.047 |
| 28 classes | 0.798 ± 0.047 | 0.798 ± 0.047 |

Paired difference on the shared structures: -0.0003 (Wilcoxon signed-rank p = 0.328).

Best five structures (45 classes): Brain-Stem 0.909, Left-Cerebellum-Cortex 0.880, Right-Cerebellum-Cortex 0.878, Right-Thalamus 0.877, Left-Thalamus 0.870

Worst five structures (45 classes): Left-choroid-plexus 0.525, WM-hypointensities 0.524, Right-vessel 0.493, Left-vessel 0.444, non-WM-hypointensities 0.101
