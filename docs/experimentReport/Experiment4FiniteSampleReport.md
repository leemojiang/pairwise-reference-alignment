# Experiment 4 Finite-Sample Behavior Report

## 1. Purpose

Experiment 4 studies how stable the alignment estimators \( \hat{A}_M \) and \( \hat{\mu}_M \) are when only a finite number of preference pairs are sampled.

This is a post-hoc analysis of the completed Qwen + RewardBench run. It does not run model inference again. Instead, it reads the saved `raw_scores.jsonl` file and repeatedly subsamples scored pairs without replacement.

This report describes Experiment 4A: a subsampling stability curve. It is not a bootstrap confidence interval analysis. The goal is to estimate how much the alignment estimate changes when only \(K\) distinct evaluation pairs are available. Experiment 4B should be used for bootstrap confidence intervals around the full-sample estimate.

For each model \(M\), each pair has a sign-agreement variable

\[
Z_k^{(M)}
=
\mathbf{1}
\left[
S_M(x_k,y_k^+) > S_M(x_k,y_k^-)
\right].
\]

For a subsample of size \(K\), the finite-sample estimator is

\[
\hat{A}_{M,K}^{(b)}
=
\frac{1}{K}
\sum_{k\in I_b}
Z_k^{(M)},
\]

where \(I_b\) is the set of sampled pair indices in repeat \(b\).

The same subsample also gives a margin estimate

\[
\hat{\mu}_{M,K}^{(b)}
=
\frac{1}{K}
\sum_{k\in I_b}
\Delta_k^{(M)}.
\]

The experiment compares empirical variability with the Hoeffding radius

\[
\epsilon_K
=
\sqrt{
\frac{1}{2K}
\log\frac{2}{\delta}
}.
\]

This run uses \(\delta=0.05\).

## 2. Run Information

Input raw scores:

```text
/pairwise-reference-alignment-code/runs/20260528-092001_experiment1and2_qwen_rewardbench_experiment12/raw_scores.jsonl
```

Experiment 4 output directory:

```text
/pairwise-reference-alignment-code/runs/20260528-102901_experiment4_qwen_rewardbench_finite_sample
```

Configuration:

- Dataset: `allenai/reward-bench`
- Models: 8 Qwen2.5 base/instruct models from 0.5B to 7B
- \(K \in \{50,100,200,500,1000,2000\}\)
- Repeats: \(B=200\)
- \(\delta=0.05\)
- Overall analysis: enabled
- Subset analysis: enabled for subsets with at least 50 examples

Output files:

- `finite_sample_overall_summary.csv`: 48 rows, one row per model and \(K\).
- `finite_sample_overall_trials.csv`: 9600 rows, one row per model, \(K\), and repeat.
- `finite_sample_subset_summary.csv`: 344 rows, subset-level summary.

## 3. Overall Pattern

The expected finite-sample behavior is observed. As \(K\) increases, the empirical standard deviation and empirical confidence interval width decrease.

For Qwen2.5-0.5B:

| \(K\) | full \(\hat{A}_M\) | sd of \(\hat{A}\) | A interval width | full \(\hat{\mu}_M\) | sd of \(\hat{\mu}\) | mu interval width |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.6148 | 0.0667 | 0.2600 | 0.0944 | 0.1208 | 0.5146 |
| 100 | 0.6148 | 0.0502 | 0.2100 | 0.0944 | 0.0857 | 0.3313 |
| 200 | 0.6148 | 0.0316 | 0.1152 | 0.0944 | 0.0568 | 0.2288 |
| 500 | 0.6148 | 0.0213 | 0.0822 | 0.0944 | 0.0354 | 0.1444 |
| 1000 | 0.6148 | 0.0137 | 0.0560 | 0.0944 | 0.0222 | 0.0876 |
| 2000 | 0.6148 | 0.0079 | 0.0330 | 0.0944 | 0.0139 | 0.0563 |

For Qwen2.5-7B-Instruct:

| \(K\) | full \(\hat{A}_M\) | sd of \(\hat{A}\) | A interval width | full \(\hat{\mu}_M\) | sd of \(\hat{\mu}\) | mu interval width |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.7705 | 0.0566 | 0.2200 | 0.3500 | 0.1285 | 0.5080 |
| 100 | 0.7705 | 0.0414 | 0.1408 | 0.3500 | 0.0968 | 0.3974 |
| 200 | 0.7705 | 0.0285 | 0.1103 | 0.3500 | 0.0655 | 0.2501 |
| 500 | 0.7705 | 0.0198 | 0.0880 | 0.3500 | 0.0415 | 0.1606 |
| 1000 | 0.7705 | 0.0113 | 0.0420 | 0.3500 | 0.0268 | 0.1059 |
| 2000 | 0.7705 | 0.0072 | 0.0265 | 0.3500 | 0.0167 | 0.0584 |

The empirical p95 absolute error for \(\hat{A}_M\) is consistently smaller than the Hoeffding radius. This is expected because Hoeffding's inequality is distribution-free and conservative. The margin estimator \(\hat{\mu}_M\) shows the same qualitative finite-sample pattern: its empirical standard deviation and interval width shrink as \(K\) grows, although it is not the bounded Bernoulli statistic used in the Hoeffding comparison.

## 4. Interpretation

The result supports the finite-sample claim in the design document:

> Larger evaluation sets produce more stable estimates of \( \hat{A}_M \), while conservative concentration bounds provide a useful but loose reference scale.

At \(K=50\), the estimator is visibly noisy. For Qwen2.5-0.5B, the empirical 95% subsampling interval width is about 0.26. This means a small pilot sample can reveal rough trends but should not be used for fine comparisons between nearby models.

At \(K=500\), the empirical subsampling interval width falls to roughly 0.08 to 0.09 in the examples above. This is already useful for distinguishing large differences, but may still be too wide for close base/instruct comparisons such as the 0.5B and 1.5B pairs.

At \(K=2000\), the empirical subsampling interval width is around 0.03. This is much more stable and is sufficient to support the larger observed gaps, such as Qwen2.5-7B-Instruct versus Qwen2.5-7B.

## 5. Relation to the Main Experiment

The main Qwen + RewardBench experiment used \(K=5120\), which is larger than the largest subsample size tested here. The finite-sample analysis therefore increases confidence that the overall ordering and margin differences observed in the main experiment are not artifacts of a very small evaluation sample.

However, subset-level results still need caution. Many RewardBench subsets are much smaller than the full dataset. For those subsets, uncertainty can remain large even when the overall result is stable.

The practical takeaway is:

- Use full RewardBench results for the main claim.
- Treat small pilot runs as debugging or trend checks.
- Treat subset-level claims as distribution-specific and sample-size-sensitive.
- Add bootstrap or finite-sample intervals to future tables or heatmaps when making fine-grained subset claims.
