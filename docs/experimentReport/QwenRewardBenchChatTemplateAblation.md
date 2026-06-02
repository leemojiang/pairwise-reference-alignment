# Qwen RewardBench Chat-Template Ablation Report

## 1. Motivation

The main Qwen RewardBench experiment used `prompt_format: plain`. This choice was intentional: it keeps the input format identical for base and instruct models, so that the comparison mainly reflects differences in model parameters rather than differences in prompt formatting.

However, Qwen2.5 tokenizers also define a chat template. In practical use, Qwen2.5-Instruct models are usually prompted with role markers and an assistant generation prefix. Therefore, this ablation uses `prompt_format: chat` to test whether the main result is sensitive to prompt construction.

The question is not whether the chat-template result should replace the plain result. Instead, the question is:

> Does the main conclusion remain true when the prompt is rendered through the Qwen tokenizer's chat template?

If the answer is yes, then the original finding is more robust. If the result changes sharply, then prompt formatting should be treated as an important experimental factor.

## 2. Experimental Setup

Plain main run:

```text
/pairwise-reference-alignment-code/runs/20260528-092001_experiment1and2_qwen_rewardbench_experiment12
```

Chat-template ablation run:

```text
/pairwise-reference-alignment-code/runs/20260528-102610_experiment1and2_qwen_rewardbench_chat_experiment12
```

SwanLab run:

```text
https://swanlab.cn/@AABBCCEEDDF/pairwise-reference-alignment/runs/ycm9ejg12fvf13jvzv1l0
```

Dataset:

```text
allenai/reward-bench
```

Models:

```text
Qwen/Qwen2.5-0.5B
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B
Qwen/Qwen2.5-7B-Instruct
```

Both runs use \(K=5120\) RewardBench preference pairs and the same scoring rule:

\[
S_M(x,y)
=
\frac{1}{|y|}
\log Q_M(y\mid x).
\]

Here \(x\) is the prompt, \(y\) is a candidate response, \(Q_M(y\mid x)\) is the conditional probability assigned by model \(M\), and \(|y|\) is the number of response tokens.

The pairwise margin is

\[
\Delta_k^{(M)}
=
S_M(x_k,y_k^+) - S_M(x_k,y_k^-),
\]

where \(y_k^+\) is the RewardBench-preferred response and \(y_k^-\) is the rejected response.

The main accuracy-like statistic is

\[
\hat{A}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}\{\Delta_k^{(M)}>0\}.
\]

The mean signed margin is

\[
\hat{\mu}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\Delta_k^{(M)}.
\]

## 3. Output Integrity

The chat-template run completed successfully.

| File | Observed content |
|---|---:|
| `raw_scores.jsonl` | 40960 rows |
| `metrics_overall.csv` | 8 model rows plus header |
| `metrics_by_subset.csv` | 184 subset rows plus header |
| `batch_timings.jsonl` | batch timing records |
| `run_meta.json` | run metadata |

The expected number of raw score rows is

\[
5120 \times 8 = 40960,
\]

so the raw output is complete. GPU memory was released after the run.

The output structure is the same as the plain Qwen + RewardBench main run:

- `config.yaml`: copied configuration for the chat-template run. The key difference from the plain run is `prompt_format: chat`.
- `run_meta.json`: reproducibility metadata, including model list, dataset list, scoring rule, seed, batch size, dtype, device, and prompt format.
- `raw_scores.jsonl`: pair-level scoring records. Each row corresponds to one `(model, dataset, scoring_rule, pair_id)` item and stores `score_chosen`, `score_rejected`, `delta`, `chosen_token_count`, `rejected_token_count`, and `subset`.
- `batch_timings.jsonl`: batch-level timing records, including elapsed seconds, processed pair count, response-token count, and tokens per second.
- `metrics_overall.csv`: one aggregate row per model. It reports the overall \(K\), \(\hat{A}_M\), \(\hat{\mu}_M\), tie rate, and average response lengths.
- `metrics_by_subset.csv`: one aggregate row per model and RewardBench subset. It reports the same aggregate fields as `metrics_overall.csv`, but restricted to each subset \(c\).

In this report, `metrics_overall.csv` is used for the plain-versus-chat comparison of \(\hat{A}_M\), \(\hat{\mu}_M\), and instruct-base gaps. `metrics_by_subset.csv` is available for later category-level prompt-format sensitivity analysis.

## 4. Overall Results

| Model | Plain \(\hat{A}_M\) | Chat \(\hat{A}_M\) | Chat - Plain |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 0.6148 | 0.6146 | -0.0002 |
| Qwen2.5-0.5B-Instruct | 0.6250 | 0.6350 | 0.0100 |
| Qwen2.5-1.5B | 0.6652 | 0.6682 | 0.0029 |
| Qwen2.5-1.5B-Instruct | 0.6715 | 0.6834 | 0.0119 |
| Qwen2.5-3B | 0.6928 | 0.6889 | -0.0039 |
| Qwen2.5-3B-Instruct | 0.7232 | 0.7717 | 0.0484 |
| Qwen2.5-7B | 0.7262 | 0.7324 | 0.0063 |
| Qwen2.5-7B-Instruct | 0.7705 | 0.7783 | 0.0078 |

The alignment accuracy pattern is mostly stable. Larger models still tend to have higher \(\hat{A}_M\), and every instruct model still has higher \(\hat{A}_M\) than its corresponding base model.

The largest change appears for Qwen2.5-3B-Instruct. Its \(\hat{A}_M\) increases from 0.7232 under the plain prompt to 0.7717 under the chat template. This suggests that the 3B-Instruct model is especially format-sensitive.

## 5. Mean Margin Results

| Model | Plain \(\hat{\mu}_M\) | Chat \(\hat{\mu}_M\) | Chat - Plain |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 0.0944 | 0.1026 | 0.0083 |
| Qwen2.5-0.5B-Instruct | 0.1177 | 0.1417 | 0.0240 |
| Qwen2.5-1.5B | 0.1388 | 0.1718 | 0.0330 |
| Qwen2.5-1.5B-Instruct | 0.1585 | 0.1947 | 0.0362 |
| Qwen2.5-3B | 0.1648 | 0.1847 | 0.0198 |
| Qwen2.5-3B-Instruct | 0.2665 | 0.5008 | 0.2343 |
| Qwen2.5-7B | 0.2005 | 0.2176 | 0.0171 |
| Qwen2.5-7B-Instruct | 0.3500 | 0.6885 | 0.3385 |

The mean margin result is more sensitive than \(\hat{A}_M\). Chat formatting increases \(\hat{\mu}_M\) for every model, and the increase is especially large for the 3B-Instruct and 7B-Instruct models.

This distinction is important. \(\hat{A}_M\) only asks whether the chosen response receives a higher score than the rejected response. \(\hat{\mu}_M\) also measures how large the signed difference is. The chat template does not merely flip many decisions; rather, it often strengthens the likelihood preference for responses that are already preferred.

## 6. Instruct-Base Gap

| Size | Plain \(\Delta \hat{A}\) | Chat \(\Delta \hat{A}\) | Plain \(\Delta \hat{\mu}\) | Chat \(\Delta \hat{\mu}\) |
|---|---:|---:|---:|---:|
| 0.5B | 0.0102 | 0.0203 | 0.0233 | 0.0390 |
| 1.5B | 0.0063 | 0.0152 | 0.0197 | 0.0228 |
| 3B | 0.0305 | 0.0828 | 0.1016 | 0.3161 |
| 7B | 0.0443 | 0.0459 | 0.1495 | 0.4709 |

The instruct-base gap remains positive at every model size. This directly supports the main conclusion of the plain experiment.

At the same time, the chat template tends to enlarge the instruct-base gap, especially for the signed margin. This is expected: instruct models are trained to operate in chat-style contexts, so the chat template may better expose the distribution learned during instruction tuning.

For this reason, the chat-template result should be interpreted as a robustness and format-sensitivity check, not as a cleaner causal comparison. The plain format remains the better-controlled comparison between base and instruct models, while the chat format is closer to normal usage of Qwen2.5-Instruct.

## 7. Diagnostic Fields

Both plain and chat runs report the same average response lengths:

| Field | Value |
|---|---:|
| `mean_chosen_len` | 241.9883 |
| `mean_rejected_len` | 188.9727 |

This is a useful sanity check. The ablation changes the rendered prompt prefix, but the measured response token lengths remain the same. Therefore, the overall comparison is not caused by a different response-length accounting scheme.

Tie rates remain low in both settings. In the chat run, tie rates range from 0.0033 to 0.0100. This means the scorer is not collapsing many chosen and rejected responses to identical values.

## 8. Interpretation

The chat-template ablation strengthens the empirical story rather than weakening it.

The main plain-prompt result was:

> Qwen2.5-Instruct models have higher likelihood-induced pairwise alignment with RewardBench preferences than their corresponding base models.

The chat-template result preserves this conclusion. In fact, it often makes the effect larger, especially for \(\hat{\mu}_M\). This suggests that the original finding is not an artifact of using a plain prompt format.

The more nuanced conclusion is:

> The likelihood-induced alignment signal is robust to Qwen chat-template prompting, but its magnitude is prompt-format sensitive.

This matters for paper writing. If the goal is a controlled base-versus-instruct comparison, the plain result should remain the primary result. If the goal is to understand how the model behaves in its intended chat interface, the chat-template result is highly relevant and should be reported as an ablation.

## 9. Preliminary Conclusion

This experiment supports the original hypothesis.

Across all four model sizes, the instruct model has higher \(\hat{A}_M\) than the corresponding base model under both prompt formats. The ordering by model scale is also broadly preserved. Therefore, the main qualitative conclusion does not depend on the plain prompt construction.

The ablation also reveals an additional empirical fact: prompt formatting affects the strength of the likelihood margin. The effect is modest for most base models but can be large for instruct models, especially Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. This suggests that future reports should always state the prompt format when reporting likelihood-induced alignment.

Recommended next steps:

1. Add a plot comparing plain and chat \(\hat{A}_M\) across model sizes.
2. Add a plot comparing plain and chat \(\hat{\mu}_M\), where the format sensitivity is more visible.
3. Inspect subset-level changes to identify which RewardBench categories are most affected by chat formatting.
4. Keep plain as the controlled main setting and report chat as a robustness or sensitivity ablation.
