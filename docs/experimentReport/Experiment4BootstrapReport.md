# Experiment 4B Bootstrap Confidence Interval Report

## 1. Purpose

Experiment 4B estimates bootstrap confidence intervals for the full-sample alignment estimators \( \hat{A}_M \) and \( \hat{\mu}_M \) from the completed Qwen + RewardBench run.

This experiment is different from Experiment 4A. Experiment 4A uses subsampling without replacement across multiple \(K\) values to study how estimates change when only \(K\) distinct evaluation pairs are available. Experiment 4B uses sampling with replacement at the full group size. It is therefore the better output to cite when reporting uncertainty around the full-sample estimate.

For a fixed group of \(K\) scored pairs, the full-sample estimator is

\[
\hat{A}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
Z_k^{(M)}.
\]

Each bootstrap repeat samples \(K\) pairs with replacement and computes

\[
\hat{A}_M^{*(b)}
=
\frac{1}{K}
\sum_{k\in I_b^*}
Z_k^{(M)}.
\]

The reported interval is the percentile interval from the bootstrap distribution of \(\hat{A}_M^{*(b)}\).

The same bootstrap sample also estimates the margin statistic \(\hat{\mu}_M^{*(b)}\), so the output contains percentile intervals for both sign agreement and signed margin.

## 2. Run Information

Input raw scores:

```text
/pairwise-reference-alignment-code/runs/20260528-092001_experiment1and2_qwen_rewardbench_experiment12/raw_scores.jsonl
```

Bootstrap output directory:

```text
/pairwise-reference-alignment-code/runs/20260528-110013_experiment4b_qwen_rewardbench_bootstrap
```

Configuration:

- Dataset: `allenai/reward-bench`
- Models: 8 Qwen2.5 base/instruct models from 0.5B to 7B
- Bootstrap repeats: \(B=1000\)
- Overall CI: enabled
- Subset CI: enabled for subsets with at least 20 examples

Output files:

- `bootstrap_overall_ci.csv`
- `bootstrap_overall_trials.csv`
- `bootstrap_subset_ci.csv`
- `config.yaml`
- `run_meta.json`

Output structure:

- `config.yaml`: copied configuration for the bootstrap run, including `raw_scores_path`, repeat count, seed, and subset filtering threshold.
- `run_meta.json`: reproducibility metadata for the bootstrap run, including the input path, number of input records, repeat count, seed, and subset settings.
- `bootstrap_overall_trials.csv`: one row per `(model, repeat)` bootstrap draw. `A_hat` and `mu_hat` are the bootstrap estimates \( \hat{A}_M^{*(b)} \) and \( \hat{\mu}_M^{*(b)} \) for that draw. `full_A_hat` and `full_mu_hat` are the original full-sample estimates repeated on each row as reference values. `K` and `full_K` are both the full group size for overall results, and `sampling` is `with_replacement`.
- `bootstrap_overall_ci.csv`: one row per model. It summarizes the bootstrap distribution from `bootstrap_overall_trials.csv`, including full-sample estimates, bootstrap means, bootstrap biases, bootstrap standard deviations, and percentile CI endpoints for both \(\hat{A}_M\) and \(\hat{\mu}_M\).
- `bootstrap_subset_ci.csv`: one row per `(model, subset)` group that passes the subset size threshold. It has the same summary fields as `bootstrap_overall_ci.csv`, but \(K\) is the subset size \(K_c\).

The most important distinction is that `A_hat` and `mu_hat` in the trials file change across bootstrap repeats, while `full_A_hat` and `full_mu_hat` are fixed within each model or subset group. The CI files are the recommended files for tables and plots; the trials file is mainly useful for drawing bootstrap distributions or diagnosing the resampling procedure.

## 3. Overall Bootstrap Confidence Intervals

| Model | \(K\) | \(\hat{A}_M\) | 95% bootstrap CI | CI width |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 5120 | 0.6148 | [0.6018, 0.6285] | 0.0268 |
| Qwen2.5-0.5B-Instruct | 5120 | 0.6250 | [0.6119, 0.6377] | 0.0258 |
| Qwen2.5-1.5B | 5120 | 0.6652 | [0.6527, 0.6779] | 0.0252 |
| Qwen2.5-1.5B-Instruct | 5120 | 0.6715 | [0.6582, 0.6850] | 0.0268 |
| Qwen2.5-3B | 5120 | 0.6928 | [0.6799, 0.7055] | 0.0256 |
| Qwen2.5-3B-Instruct | 5120 | 0.7232 | [0.7113, 0.7346] | 0.0232 |
| Qwen2.5-7B | 5120 | 0.7262 | [0.7137, 0.7383] | 0.0246 |
| Qwen2.5-7B-Instruct | 5120 | 0.7705 | [0.7590, 0.7822] | 0.0232 |

The overall intervals are fairly tight because the full RewardBench run has \(K=5120\) pairs. The bootstrap bias is also close to zero for all models, which is expected for this simple mean estimator.

The margin bootstrap intervals are wider in absolute scale but show the same main pattern:

| Model | \(K\) | \(\hat{\mu}_M\) | 95% bootstrap CI | CI width |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 5120 | 0.0944 | [0.0707, 0.1169] | 0.0462 |
| Qwen2.5-0.5B-Instruct | 5120 | 0.1177 | [0.0926, 0.1402] | 0.0476 |
| Qwen2.5-1.5B | 5120 | 0.1388 | [0.1168, 0.1596] | 0.0428 |
| Qwen2.5-1.5B-Instruct | 5120 | 0.1585 | [0.1349, 0.1826] | 0.0477 |
| Qwen2.5-3B | 5120 | 0.1648 | [0.1447, 0.1848] | 0.0400 |
| Qwen2.5-3B-Instruct | 5120 | 0.2665 | [0.2362, 0.2961] | 0.0599 |
| Qwen2.5-7B | 5120 | 0.2005 | [0.1817, 0.2205] | 0.0388 |
| Qwen2.5-7B-Instruct | 5120 | 0.3500 | [0.3228, 0.3775] | 0.0547 |

## 4. Interpretation

The bootstrap intervals strengthen the main experiment's interpretation.

The large model comparisons remain clearly separated. For example, Qwen2.5-7B has \(\hat{A}_M=0.7262\) with CI `[0.7137, 0.7383]`, while Qwen2.5-7B-Instruct has \(\hat{A}_M=0.7705\) with CI `[0.7590, 0.7822]`. These intervals do not overlap, so the 7B instruct improvement is unlikely to be explained by finite-sample noise alone. The margin intervals also do not overlap for this comparison: Qwen2.5-7B has \(\hat{\mu}_M=0.2005\) with CI `[0.1817, 0.2205]`, while Qwen2.5-7B-Instruct has \(\hat{\mu}_M=0.3500\) with CI `[0.3228, 0.3775]`.

The 3B comparison is also clearly separated for both metrics. For \(\hat{A}_M\), Qwen2.5-3B has CI `[0.6799, 0.7055]`, while Qwen2.5-3B-Instruct has CI `[0.7113, 0.7346]`. For \(\hat{\mu}_M\), Qwen2.5-3B has CI `[0.1447, 0.1848]`, while Qwen2.5-3B-Instruct has CI `[0.2362, 0.2961]`.

The smaller model comparisons are less decisive. The 0.5B base and instruct intervals overlap, and the 1.5B base and instruct intervals also overlap. This matches the original effect sizes: the observed instruct gain is much smaller for 0.5B and 1.5B.

Therefore, the current evidence should be phrased as:

> In the Qwen2.5 family on RewardBench, instruction tuning shows a robust likelihood-induced alignment improvement for 3B and 7B models. For 0.5B and 1.5B models, the direction is positive but the bootstrap intervals overlap, so the evidence is weaker.

## 5. Subset-Level Caution

Subset-level bootstrap intervals are much wider than the overall intervals. This is expected because each subset has a smaller \(K_c\).

For example, in the Qwen2.5-0.5B results:

- `alpacaeval-easy` has \(K_c=805\) and CI width about 0.067.
- `donotanswer` has \(K_c=136\) and CI width about 0.154.
- `llmbar-adver-GPTInst` has \(K_c=92\) and CI width about 0.174.

This supports the caution already raised in Experiment 4A: subset heatmaps are useful, but small-subset differences should not be overinterpreted without uncertainty intervals.

## 6. Reporting Recommendation

For the main table, report both \( \hat{A}_M \) and \( \hat{\mu}_M \) together with the overall bootstrap CIs from `bootstrap_overall_ci.csv`.

For subset heatmaps, either:

1. show only subsets with sufficiently large \(K_c\), or
2. add bootstrap CI width as an auxiliary heatmap or table.

Experiment 4A and Experiment 4B should be described as complementary:

- Experiment 4A: how estimate stability changes with the number of distinct evaluated pairs.
- Experiment 4B: bootstrap uncertainty around the full-sample reported estimate.

