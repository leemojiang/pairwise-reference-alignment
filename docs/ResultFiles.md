# Experiment Result Files

本文档解释实验脚本生成的结果文件，以及每个指标如何对应实验设计文档中的定义。

实验设计中的主目标是检验：在固定 reference preference pair 上，语言模型自身的 token-normalized likelihood ordering 是否与 reference preference ordering 一致。实验 1 关注 overall alignment，实验 2 关注 subset-level alignment map。

## Run 目录

每次运行脚本都会在 `output_dir` 下创建一个新的 run 目录，例如：

```text
runs/20260527-120000_experiment1_qwen_rewardbench_experiment1/
```

常见文件包括：

- `config.yaml`：本次运行使用的配置副本。
- `run_meta.json`：本次运行的元信息，例如模型列表、数据集列表、batch size、dtype、device、seed。
- `raw_scores.jsonl`：逐个 preference pair 保存的原始 scoring 结果。
- `batch_timings.jsonl`：逐个 batch 保存的运行时间和吞吐。
- `metrics_overall.csv`：overall 聚合指标，主要对应实验 1。
- `metrics_by_subset.csv`：subset-level 聚合指标，主要对应实验 2。
- `verify_env.json`：环境检查结果，由 `verify_env.py` 生成。
- `cache_env.json` 和 `cache_items.jsonl`：缓存检查和预下载结果，由 `prepare_cache.py` 生成。

## Reference Pair

实验数据由 preference pairs 组成。第 \(k\) 个样本记为：

\[
(x_k,y_k^+,y_k^-).
\]

其中：

- \(x_k\)：prompt；
- \(y_k^+\)：reference-preferred response，也就是数据集中标记为 preferred/chosen 的回答；
- \(y_k^-\)：reference-rejected response，也就是数据集中标记为 rejected 的回答；
- \(k=1,\ldots,K\)，其中 \(K\) 是当前 evaluation set 中 pair 的数量。

数据集只提供 reference preference，不引入 reward model。

## Scoring Rule

当前主实验使用 token-normalized log-likelihood。给定模型 \(M\)、prompt \(x\) 和 response \(y\)，score 定义为：

\[
S_M(x,y)
=
\frac{1}{|y|}
\log Q_M(y\mid x).
\]

其中：

- \(Q_M(y\mid x)\)：模型 \(M\) 在 prompt \(x\) 条件下生成 response \(y\) 的条件概率；
- \(|y|\)：response \(y\) 的 token 数；
- \(S_M(x,y)\)：token-normalized log-likelihood score。

使用 token normalization 是为了减弱 response length 对 raw sequence likelihood 的影响。

## raw_scores.jsonl

`raw_scores.jsonl` 是最重要的中间结果文件。每一行对应一个模型在一个 pair 上的 scoring 结果。

示例：

```json
{
  "pair_id": "pair_000001",
  "subset": "chat",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "dataset": "allenai/reward-bench",
  "score_chosen": -1.82,
  "score_rejected": -2.14,
  "delta": 0.32,
  "chosen_token_count": 128,
  "rejected_token_count": 147,
  "scoring_rule": "token_normalized_loglikelihood"
}
```

字段解释：

- `pair_id`：样本 ID，用于追踪具体 preference pair。
- `subset`：样本所属 subset/category，例如 chat、safety、reasoning。如果数据集没有明确字段，代码会使用可识别的 category/source 字段，或回退为 `overall`。
- `model`：当前 scoring 使用的模型。
- `dataset`：当前样本来自的数据集。
- `score_chosen`：\(S_M(x_k,y_k^+)\)，模型对 reference-preferred response 的 token-normalized log-likelihood。
- `score_rejected`：\(S_M(x_k,y_k^-)\)，模型对 reference-rejected response 的 token-normalized log-likelihood。
- `delta`：二者差值。

\[
\Delta_k^{(M)}
=
S_M(x_k,y_k^+)-S_M(x_k,y_k^-).
\]

- `chosen_token_count`：计算 \(S_M(x_k,y_k^+)\) 时 response 部分的 token 数。
- `rejected_token_count`：计算 \(S_M(x_k,y_k^-)\) 时 response 部分的 token 数。
- `scoring_rule`：当前使用的 scoring rule。

如果 \(\Delta_k^{(M)}>0\)，说明模型 \(M\) 对该 pair 的 likelihood ordering 与 reference preference 方向一致。如果 \(\Delta_k^{(M)}<0\)，说明模型更偏向 rejected response。

## metrics_overall.csv

`metrics_overall.csv` 是对 `raw_scores.jsonl` 的 overall 聚合。它主要对应实验设计中的 Priority 1: Likelihood-Induced Pairwise Alignment。

每一行通常对应一个 `(model, dataset, subset="overall")`。

核心字段：

- `model`：模型名称。
- `dataset`：数据集名称。
- `subset`：overall 表中固定为 `overall`。
- `K`：参与聚合的 pair 数量。
- `A_hat`：alignment accuracy，对应实验设计中的 \(\hat{A}_M\)。
- `mu_hat`：mean signed margin，对应实验设计中的 \(\hat{\mu}_M\)。

实验设计中的 sign agreement indicator 为：

\[
Z_M(x_k,y_k^+,y_k^-)
=
\mathbf{1}
\left[
S_M(x_k,y_k^+) > S_M(x_k,y_k^-)
\right].
\]

因此：

\[
\hat{A}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
Z_M(x_k,y_k^+,y_k^-).
\]

\(\hat{A}_M\) 越高，说明模型的 likelihood ordering 越常与 reference preference 一致。

margin estimator 为：

\[
\hat{\mu}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\left[
S_M(x_k,y_k^+) - S_M(x_k,y_k^-)
\right].
\]

\(\hat{\mu}_M\) 不只看方向，也看 signed score strength。若 \(\hat{\mu}_M>0\)，说明平均意义上模型给 preferred response 更高 likelihood score。

当前代码没有直接输出 \(\hat{m}_M^{\mathrm{sign}}\)，但可以由 `A_hat` 得到：

\[
\hat{m}_M^{\mathrm{sign}}
=
2\hat{A}_M-1.
\]

## metrics_by_subset.csv

`metrics_by_subset.csv` 是对每个 subset 单独聚合的结果。它主要对应实验设计中的 Priority 2: Subset-Level Alignment Map。

如果第 \(k\) 个 pair 属于 subset \(c\)，则 subset-level alignment 为：

\[
\hat{A}_M^{(c)}
=
\frac{1}{K_c}
\sum_{k:c_k=c}
\mathbf{1}
\left[
S_M(x_k,y_k^+) > S_M(x_k,y_k^-)
\right],
\]

其中 \(K_c\) 是 subset \(c\) 中的 pair 数量。

subset-level margin 为：

\[
\hat{\mu}_M^{(c)}
=
\frac{1}{K_c}
\sum_{k:c_k=c}
\left[
S_M(x_k,y_k^+) - S_M(x_k,y_k^-)
\right].
\]

这个文件用于观察：同一个模型是否在 chat、safety、reasoning 等不同 pair distribution 上表现不同。

## Diagnostic Fields

`tie_rate`、`mean_chosen_len`、`mean_rejected_len` 不是 Priority 1 主表中的核心理论估计量。它们是代码额外保存的诊断字段，用于帮助解释和审查 \(\hat{A}_M\)、\(\hat{\mu}_M\) 是否可靠。

### tie_rate

`tie_rate` 表示 score 完全相等的 pair 占比：

\[
\mathrm{tie\_rate}
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}
\left[
S_M(x_k,y_k^+) = S_M(x_k,y_k^-)
\right].
\]

在浮点 likelihood scoring 中，完全相等通常不常见。如果 `tie_rate` 很高，可能说明：

- mock scorer 或某个异常 scorer 输出了大量相同分数；
- response token mask 出错，导致 chosen/rejected 实际 scoring 内容为空或相同；
- 数据集中存在大量重复 chosen/rejected；
- 分数被过度 round 或后处理。

因此 `tie_rate` 不是 alignment 指标本身，而是结果质量检查字段。

### mean_chosen_len

`mean_chosen_len` 是 preferred response 的平均 token 数：

\[
\mathrm{mean\_chosen\_len}
=
\frac{1}{K}
\sum_{k=1}^{K}
|y_k^+|.
\]

其中 \(|y_k^+|\) 是模型 tokenizer 下 chosen response 的 token 数。

这个字段用于检查 length effect。虽然主 score 已经做了 token normalization，但如果 chosen 和 rejected 的长度分布差异很大，仍然需要在解释结果时谨慎。

### mean_rejected_len

`mean_rejected_len` 是 rejected response 的平均 token 数：

\[
\mathrm{mean\_rejected\_len}
=
\frac{1}{K}
\sum_{k=1}^{K}
|y_k^-|.
\]

其中 \(|y_k^-|\) 是模型 tokenizer 下 rejected response 的 token 数。

如果 `mean_rejected_len` 明显大于 `mean_chosen_len`，或者反过来，说明 reference preference pairs 可能带有长度偏差。此时应同时查看 `mu_hat` 和 `A_hat`，不要只凭一个指标下结论。

## batch_timings.jsonl

`batch_timings.jsonl` 记录每个 batch 的运行效率。

常见字段：

- `batch_index`：batch 编号。
- `elapsed_seconds`：该 batch 用时。
- `pairs`：该 batch 中处理的 preference pair 数量。
- `tokens`：该 batch 中 response token 的数量。
- `tokens_per_second`：吞吐量。
- `dataset`、`model`、`scoring_rule`：当前 batch 所属设置。

这个文件主要用于判断 batch size 是否合适、是否存在异常慢 batch，以及不同模型之间的运行成本差异。

## run_meta.json

`run_meta.json` 保存复现实验所需的运行元信息，例如：

- `experiment_id`；
- `models`；
- `datasets`；
- `scoring_rules`；
- `limit`；
- `batch_size`；
- `max_length`；
- `dtype`；
- `device`；
- `seed`。

写论文或复现实验时，应优先查看 `config.yaml` 和 `run_meta.json`，确认当前表格来自哪一次设置。

## Interpreting Results

Priority 1 的主要判断方式：

- 比较同一家族的 base 和 instruct model；
- 如果 instruct model 的 `A_hat` 和 `mu_hat` 通常高于 base model，说明 instruction/preference tuning 可能在模型 likelihood distribution 中留下了可测的 ordinal trace；
- 如果某些模型没有提升，不一定是坏结果，可能说明 likelihood-induced alignment 是 distribution-dependent 的。

Priority 2 的主要判断方式：

- 查看 `metrics_by_subset.csv`；
- 比较每个模型在不同 subset 上的 `A_hat` 和 `mu_hat`；
- 如果模型 overall 接近，但 subset 表现不同，说明 alignment claim 与 reference pair distribution 有关。

诊断字段的使用方式：

- `tie_rate` 高时，先检查 scoring 或 mask，而不是直接解释为模型没有偏好；
- `mean_chosen_len` 与 `mean_rejected_len` 差异大时，要在结果分析中说明 length distribution；
- subset 的 `K` 很小时，该 subset 的估计不稳定，应避免过度解释。

## Minimal Reading Order

分析一个 run 时，建议按下面顺序阅读：

1. `config.yaml`：确认实验设置。
2. `run_meta.json`：确认 seed、batch size、dtype、device。
3. `batch_timings.jsonl`：确认运行过程是否正常。
4. `raw_scores.jsonl`：抽查 score 和 delta 方向。
5. `metrics_overall.csv`：分析 Priority 1。
6. `metrics_by_subset.csv`：分析 Priority 2。
