# Experiment 4B: Bootstrap Confidence Intervals

## Purpose

Experiment 4B estimates bootstrap confidence intervals for the full-sample alignment estimators \( \hat{A}_M \) and \( \hat{\mu}_M \).

This is different from Experiment 4A. Experiment 4A uses subsampling without replacement across multiple \(K\) values to ask how stable the estimator would be if only \(K\) distinct evaluation pairs were available. Experiment 4B uses sampling with replacement at the full group size to estimate uncertainty around the observed full-sample estimate.

For a fixed group of \(K\) scored preference pairs, the full-sample estimator is

\[
\hat{A}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
Z_k^{(M)}.
\]

Each bootstrap repeat samples \(K\) pairs with replacement from the same group and computes

\[
\hat{A}_M^{*(b)}
=
\frac{1}{K}
\sum_{k\in I_b^*}
Z_k^{(M)},
\]

where \(I_b^*\) is a bootstrap sample that may contain repeated pair indices.

The same bootstrap sample also computes the margin estimator

\[
\hat{\mu}_M^{*(b)}
=
\frac{1}{K}
\sum_{k\in I_b^*}
\Delta_k^{(M)}.
\]

## Recommended Main Configuration

Run:

```bash
python scripts/run_experiment4b.py --config configs/experiment4b_qwen_rewardbench_bootstrap.yaml
```

The default configuration uses:

- `raw_scores_path`: the full Qwen + RewardBench `raw_scores.jsonl` file.
- `repeats: 1000`.
- Overall bootstrap CI for each model.
- Subset-level bootstrap CI for subsets with at least 20 examples.

## Outputs

Each run creates a new directory under `runs/`.

Main files:

- `bootstrap_overall_trials.csv`: one row per overall bootstrap repeat, including `A_hat` and `mu_hat`.
- `bootstrap_overall_ci.csv`: bootstrap confidence intervals for each model, including both \(\hat{A}_M\) and \(\hat{\mu}_M\).
- `bootstrap_subset_ci.csv`: bootstrap confidence intervals for each model and subset.
- `config.yaml`: copied configuration.
- `run_meta.json`: metadata for reproducibility.

## Summary Columns

The CI files contain:

- `K`: bootstrap sample size, equal to the full group size.
- `full_K`: number of scored pairs in the group.
- `full_A_hat`: full-sample estimate.
- `full_mu_hat`: full-sample margin estimate.
- `repeats`: number of bootstrap repetitions.
- `bootstrap_mean_A_hat`: mean of bootstrap estimates.
- `bootstrap_bias`: `bootstrap_mean_A_hat - full_A_hat`.
- `bootstrap_sd_A_hat`: empirical standard deviation of bootstrap estimates.
- `bootstrap_ci_low_2p5` and `bootstrap_ci_high_97p5`: percentile bootstrap interval endpoints.
- `bootstrap_ci_width_95`: width of the percentile interval.
- `bootstrap_mean_mu_hat`, `bootstrap_mu_bias`, `bootstrap_sd_mu_hat`, `bootstrap_mu_ci_low_2p5`, `bootstrap_mu_ci_high_97p5`, and `bootstrap_mu_ci_width_95`: the corresponding bootstrap statistics for \(\hat{\mu}_M\).

## Interpretation

Use Experiment 4B when reporting uncertainty around the full-sample result, for example:

```text
Qwen2.5-7B-Instruct: A_hat = 0.7705, 95% bootstrap CI [low, high].
```

Use Experiment 4A when discussing how many distinct evaluation pairs are needed before estimates become stable.
