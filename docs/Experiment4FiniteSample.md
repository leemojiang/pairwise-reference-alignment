# Experiment 4: Finite-Sample Behavior

## Purpose

Experiment 4 connects the finite-sample concentration discussion in the experiment design with the observed stability of the estimator \( \hat{A}_M \).

Unlike Experiments 1 and 2, this experiment does not run model inference. It reads an existing `raw_scores.jsonl` file and repeatedly subsamples already-scored preference pairs without replacement.

This is a subsampling stability analysis, not a bootstrap confidence interval analysis. Its purpose is to ask how unstable \(\hat{A}_M\) would be if the evaluation only contained \(K\) distinct pairs. Bootstrap confidence intervals answer a related but different question and are handled separately in Experiment 4B.

For each model \(M\), each preference pair has a binary sign-agreement value

\[
Z_k^{(M)}
=
\mathbf{1}
\left[
S_M(x_k,y_k^+) > S_M(x_k,y_k^-)
\right].
\]

For a subsample of size \(K\), the estimator is

\[
\hat{A}_{M,K}
=
\frac{1}{K}
\sum_{k=1}^{K}
Z_k^{(M)}.
\]

The script repeats this subsampling \(B\) times for each \(K\), then reports the empirical variability of \(\hat{A}_{M,K}\).

## Recommended Main Configuration

The Qwen + RewardBench finite-sample analysis uses:

```bash
python scripts/run_experiment4.py --config configs/experiment4_qwen_rewardbench_finite_sample.yaml
```

The default configuration uses:

- `raw_scores_path`: the full Qwen + RewardBench `raw_scores.jsonl` file.
- \(K \in \{50,100,200,500,1000,2000\}\).
- \(B=200\) repeated subsamples.
- \(\delta=0.05\) for the Hoeffding radius.
- Overall analysis for every model.
- Subset-level analysis for subsets with at least 50 examples.

## Outputs

Each run creates a new directory under `runs/`.

Main files:

- `finite_sample_overall_trials.csv`: one row per repeated overall subsample, including both `A_hat` and `mu_hat`.
- `finite_sample_overall_summary.csv`: aggregated finite-sample statistics for each model and \(K\), including both sign accuracy and margin variability.
- `finite_sample_subset_summary.csv`: aggregated subset-level finite-sample statistics.
- `config.yaml`: copied configuration.
- `run_meta.json`: metadata for reproducibility.

## Summary Columns

`finite_sample_overall_summary.csv` and `finite_sample_subset_summary.csv` contain:

- `K`: subsample size.
- `full_K`: number of available scored pairs in the full group.
- `full_A_hat`: \(\hat{A}_M\) computed on the full available group.
- `full_mu_hat`: \(\hat{\mu}_M\) computed on the full available group.
- `repeats`: number of subsampling repetitions.
- `mean_A_hat`: mean of subsampled \(\hat{A}_{M,K}\) across repetitions.
- `sd_A_hat`: empirical standard deviation across repetitions.
- `ci_low_2p5` and `ci_high_97p5`: 2.5% and 97.5% quantiles of the repeated subsampling estimates. These columns describe an empirical subsampling interval, not a bootstrap confidence interval.
- `ci_width_95`: width of that empirical subsampling interval.
- `empirical_abs_error_p95`: 95th percentile of \(|\hat{A}_{M,K}^{(b)}-\hat{A}_M^{\mathrm{full}}|\).
- `mean_mu_hat`, `sd_mu_hat`, `mu_ci_low_2p5`, `mu_ci_high_97p5`, `mu_ci_width_95`, and `mu_empirical_abs_error_p95`: the corresponding subsampling stability statistics for \(\hat{\mu}_M\).
- `hoeffding_radius`: conservative Hoeffding radius

\[
\epsilon_K
=
\sqrt{
\frac{1}{2K}
\log\frac{2}{\delta}
}.
\]

- `hoeffding_coverage_vs_full`: fraction of repetitions where the subsample estimate is within `hoeffding_radius` of the full-sample estimate.

## Interpretation

If the experiment behaves as expected:

- `sd_A_hat` decreases as \(K\) increases.
- `ci_width_95` decreases as \(K\) increases.
- `sd_mu_hat` and `mu_ci_width_95` also decrease as \(K\) increases, although \(\hat{\mu}_M\) does not have the same bounded Bernoulli interpretation as \(\hat{A}_M\).
- `hoeffding_radius` is larger than the empirical error scale, because it is a conservative distribution-free bound.
- Small subsets show larger uncertainty than the overall RewardBench estimate.

This experiment should be reported as an uncertainty and reliability analysis, not as a new alignment result. Its main role is to show how much sample size is needed before the observed ordering of models becomes stable.

When reporting uncertainty around the full-sample estimate itself, use the bootstrap outputs from Experiment 4B rather than treating these subsampling intervals as formal confidence intervals.
