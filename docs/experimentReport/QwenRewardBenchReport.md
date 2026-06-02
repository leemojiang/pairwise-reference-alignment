# Qwen RewardBench Experiment Report

## 1. Experiment Overview

This report summarizes the first full main experiment for the pairwise reference alignment project. The experiment evaluates whether a language model's own token-normalized likelihood can recover the preference ordering provided by RewardBench.

For each preference pair, we write the sample as

\[
(x_k, y_k^+, y_k^-),
\]

where \(x_k\) is the prompt, \(y_k^+\) is the reference-preferred response, \(y_k^-\) is the reference-rejected response, and \(k=1,\ldots,K\). In this run, \(K=5120\) for RewardBench.

For a model \(M\), the score used in the experiment is token-normalized log-likelihood:

\[
S_M(x,y)
=
\frac{1}{|y|}
\log Q_M(y\mid x).
\]

Here \(Q_M(y\mid x)\) is the conditional probability assigned by model \(M\) to response \(y\) given prompt \(x\), and \(|y|\) is the number of response tokens under the model tokenizer.

The pairwise margin is

\[
\Delta_k^{(M)}
=
S_M(x_k,y_k^+) - S_M(x_k,y_k^-).
\]

If \(\Delta_k^{(M)}>0\), then model \(M\)'s likelihood ordering agrees with the reference preference for pair \(k\).

The main overall estimators are:

\[
\hat{A}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}\{\Delta_k^{(M)}>0\},
\]

and

\[
\hat{\mu}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\Delta_k^{(M)}.
\]

\(\hat{A}_M\) measures the fraction of pairs where the model gives higher likelihood to the preferred response. \(\hat{\mu}_M\) measures the average signed strength of this preference.

## 2. Run Configuration

Run directory:

```text
/pairwise-reference-alignment-code/runs/20260528-092001_experiment1and2_qwen_rewardbench_experiment12
```

Main output files:

- `metrics_overall.csv`: 8 rows, one overall result for each model.
- `metrics_by_subset.csv`: 184 subset result rows plus header, covering 23 RewardBench subsets for each of 8 models.
- `raw_scores.jsonl`: 40960 rows, corresponding to \(5120\) pairs times 8 models.
- `batch_timings.jsonl`: 20480 rows, corresponding to batch-level timing records with `batch_size=2`.
- `run_meta.json`: run metadata including seed, models, dataset, device, dtype, and batch size.

Models:

| Model | Type | Size |
|---|---|---:|
| Qwen/Qwen2.5-0.5B | base | 0.5B |
| Qwen/Qwen2.5-0.5B-Instruct | instruct | 0.5B |
| Qwen/Qwen2.5-1.5B | base | 1.5B |
| Qwen/Qwen2.5-1.5B-Instruct | instruct | 1.5B |
| Qwen/Qwen2.5-3B | base | 3B |
| Qwen/Qwen2.5-3B-Instruct | instruct | 3B |
| Qwen/Qwen2.5-7B | base | 7B |
| Qwen/Qwen2.5-7B-Instruct | instruct | 7B |

Dataset:

```text
allenai/reward-bench
```

Scoring rule:

```text
token_normalized_loglikelihood
```

Prompt format:

```text
plain
```

The same plain prompt construction is used for base and instruct models. This avoids mixing two factors: model training difference and chat-template formatting difference.

## 3. Overall Results

| Model | \(K\) | \(\hat{A}_M\) | \(\hat{\mu}_M\) | tie rate |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 5120 | 0.6148 | 0.0944 | 0.0078 |
| Qwen2.5-0.5B-Instruct | 5120 | 0.6250 | 0.1177 | 0.0057 |
| Qwen2.5-1.5B | 5120 | 0.6652 | 0.1388 | 0.0068 |
| Qwen2.5-1.5B-Instruct | 5120 | 0.6715 | 0.1585 | 0.0064 |
| Qwen2.5-3B | 5120 | 0.6928 | 0.1648 | 0.0055 |
| Qwen2.5-3B-Instruct | 5120 | 0.7232 | 0.2665 | 0.0057 |
| Qwen2.5-7B | 5120 | 0.7262 | 0.2005 | 0.0041 |
| Qwen2.5-7B-Instruct | 5120 | 0.7705 | 0.3500 | 0.0023 |

The overall result shows two clear trends.

First, larger Qwen2.5 models tend to have higher likelihood-induced pairwise alignment. The base models improve from \(\hat{A}_M=0.6148\) for Qwen2.5-0.5B to \(\hat{A}_M=0.7262\) for Qwen2.5-7B. The instruct models improve from \(\hat{A}_M=0.6250\) for Qwen2.5-0.5B-Instruct to \(\hat{A}_M=0.7705\) for Qwen2.5-7B-Instruct.

Second, within each model size, the instruct model has higher \(\hat{A}_M\) and higher \(\hat{\mu}_M\) than the corresponding base model.

| Pair | \(\Delta \hat{A}\) | \(\Delta \hat{\mu}\) |
|---|---:|---:|
| 0.5B-Instruct minus 0.5B | 0.0102 | 0.0233 |
| 1.5B-Instruct minus 1.5B | 0.0063 | 0.0197 |
| 3B-Instruct minus 3B | 0.0305 | 0.1016 |
| 7B-Instruct minus 7B | 0.0443 | 0.1495 |

The instruction-tuning gain is small for 0.5B and 1.5B, but becomes larger for 3B and 7B. This is consistent with the idea that post-training can leave a measurable ordinal trace in the model's own likelihood distribution, especially when the model has enough capacity.

## 4. Relation to the Hypothesis

The design hypothesis was:

> If instruction tuning or preference tuning changes the model distribution so that reference-preferred responses become more natural under the model, then the tuned model should generally have higher \(\hat{A}_M\) and \(\hat{\mu}_M\) than the corresponding base model.

The current result is broadly consistent with this hypothesis.

The evidence is strongest in the 3B and 7B comparisons. For Qwen2.5-7B, instruction tuning increases \(\hat{A}_M\) from 0.7262 to 0.7705, and increases \(\hat{\mu}_M\) from 0.2005 to 0.3500. This means the instruct model not only agrees with the reference preference on more pairs, but also gives a stronger average signed likelihood margin to the preferred responses.

The result also supports the weaker claim that raw generative likelihood contains nontrivial preference information. Even the base models have \(\hat{A}_M>0.5\), which means their likelihood ordering is better than random with respect to RewardBench preferences.

However, this experiment should not be interpreted as proving that Qwen2.5-Instruct was trained on RewardBench-like preferences in a direct causal sense. RewardBench is a broad benchmark, and Qwen Instruct training data is not controlled here. The more careful conclusion is:

> In this model family and dataset, instruction-tuned models show stronger likelihood-induced alignment with RewardBench preference pairs than their base counterparts.

This is already useful for the paper's main empirical story, because the proposed observable is not merely noise. It can distinguish model scale, base versus instruct variants, and subset-level differences.

## 5. Subset-Level Results

The subset-level file contains 23 RewardBench subsets for each of 8 models. The results show that likelihood-induced alignment is highly distribution-dependent.

For nearly all models, the strongest subsets are HEP/code-related subsets. For example:

| Model | High-performing subsets by \(\hat{A}_M^{(c)}\) |
|---|---|
| Qwen2.5-0.5B | `hep-java`, `hep-python`, `hep-js`, `hep-go`, `hep-cpp` |
| Qwen2.5-1.5B | `hep-java`, `hep-js`, `hep-python`, `hep-cpp`, `hep-go` |
| Qwen2.5-3B | `hep-java`, `hep-python`, `hep-cpp`, `hep-js`, `hep-go` |
| Qwen2.5-7B | `hep-java`, `hep-python`, `hep-cpp`, `hep-go`, `hep-js` |

The weakest subsets are usually adversarial, refusal, or safety-related subsets. Common low-performing subsets include:

```text
llmbar-adver-GPTInst
llmbar-adver-GPTOut
llmbar-adver-manual
donotanswer
refusals-dangerous
xstest-should-refuse
```

This pattern matters. The overall score alone might suggest a single alignment level for each model, but the subset results show that the observable depends strongly on the reference pair distribution \(P_{\mathrm{pair}}^{(c)}\). This matches the Priority 2 motivation in the experiment design: alignment claims should be interpreted relative to the subset or pair distribution being evaluated.

## 6. 7B Base vs 7B Instruct by Subset

The largest 7B instruct improvements appear in:

| Subset | \(K_c\) | 7B base \(\hat{A}^{(c)}\) | 7B instruct \(\hat{A}^{(c)}\) | Difference |
|---|---:|---:|---:|---:|
| `mt-bench-easy` | 28 | 0.500 | 0.679 | 0.179 |
| `xstest-should-refuse` | 154 | 0.331 | 0.500 | 0.169 |
| `alpacaeval-easy` | 805 | 0.652 | 0.802 | 0.150 |
| `donotanswer` | 136 | 0.309 | 0.404 | 0.096 |
| `alpacaeval-hard` | 803 | 0.834 | 0.894 | 0.060 |

The largest decreases for 7B instruct compared with 7B base are smaller in magnitude:

| Subset | \(K_c\) | 7B base \(\hat{A}^{(c)}\) | 7B instruct \(\hat{A}^{(c)}\) | Difference |
|---|---:|---:|---:|---:|
| `refusals-dangerous` | 100 | 0.370 | 0.340 | -0.030 |
| `mt-bench-hard` | 45 | 0.711 | 0.689 | -0.022 |
| `llmbar-adver-neighbor` | 134 | 0.582 | 0.567 | -0.015 |
| `hep-go` | 164 | 0.970 | 0.957 | -0.012 |
| `hep-rust` | 164 | 0.927 | 0.915 | -0.012 |

This suggests that instruction tuning improves the overall alignment signal, but the gain is uneven. It is especially visible in some general chat and refusal-related subsets, while some adversarial or safety subsets remain difficult.

One important caution is that several subsets have small \(K_c\), such as `mt-bench-easy` with \(K_c=28\) and `mt-bench-hard` with \(K_c=45\). Their subset estimates can be noisy. They should be treated as exploratory signals unless later accompanied by confidence intervals or bootstrap analysis.

## 7. Diagnostic Fields

The tie rates are low for all models, from 0.0023 to 0.0078. This is a good sanity check: the scorer is not producing a large number of identical chosen and rejected scores.

The average token lengths are:

| Field | Value |
|---|---:|
| `mean_chosen_len` | 241.9883 |
| `mean_rejected_len` | 188.9727 |

The preferred responses are longer on average than the rejected responses. Because the score uses token-normalized log-likelihood, the most direct raw sequence-length bias is reduced. Still, this length imbalance should be mentioned when interpreting the result. A future analysis should check whether \(\Delta_k^{(M)}\) remains stable after stratifying by response length difference.

## 8. Preliminary Conclusion

The experiment provides a positive first result for the project.

It supports the main Priority 1 hypothesis: the likelihood-induced pairwise alignment observable can distinguish base and instruct models within the same Qwen2.5 family. In every size tested, the instruct model has higher \(\hat{A}_M\) and higher \(\hat{\mu}_M\) than the corresponding base model.

It also supports the Priority 2 hypothesis: alignment is not a single global property independent of data distribution. The subset-level results vary widely. HEP/code subsets are very high, while adversarial, refusal, and safety-related subsets are much weaker. Therefore, future reports should avoid saying simply "model \(M\) is aligned" and should instead say "model \(M\) has alignment level \(\hat{A}_M\) under this reference pair distribution."

The current result is best framed as initial empirical evidence, not as a final causal claim about Qwen's training data. The next most useful steps are:

1. Add confidence intervals or bootstrap standard errors for \(\hat{A}_M\) and \(\hat{\mu}_M\).
2. Plot subset heatmaps for \(\hat{A}_M^{(c)}\) and \(\hat{\mu}_M^{(c)}\).
3. Compare with a preference-specific model pair, such as Mistral base versus Zephyr, if access and compute allow.
4. Run a robustness experiment on another preference dataset, such as HH-RLHF.
5. Analyze whether response length difference explains part of the margin signal.

