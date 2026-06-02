# Experiment Execution Plan

本文档给出当前代码版本推荐执行的实验层次。**第一阶段只执行 Qwen + RewardBench 主线**。Zephyr/Mistral 和 HH-RLHF 相关配置保留为后续扩展，不在第一阶段运行。

实验 1 和实验 2 已经在代码层面合并为一次 scoring run：同一次运行会生成 `metrics_overall.csv` 和 `metrics_by_subset.csv`。论文或分析中仍然可以把二者分开叙述：

- Experiment 1: 使用 `metrics_overall.csv` 分析 overall likelihood-induced pairwise alignment。
- Experiment 2: 使用 `metrics_by_subset.csv` 分析 subset-level alignment map。

## Shared Setup

所有主实验使用同一个 scoring rule：

\[
S_M(x,y)
=
\frac{1}{|y|}
\log Q_M(y\mid x).
\]

其中：

- \(M\)：被评估的 generative language model；
- \(x\)：prompt；
- \(y\)：response；
- \(|y|\)：response 在当前 tokenizer 下的 token 数；
- \(S_M(x,y)\)：token-normalized log-likelihood score。

对于每个 preference pair：

\[
(x_k,y_k^+,y_k^-),
\]

定义：

\[
\Delta_k^{(M)}
=
S_M(x_k,y_k^+)-S_M(x_k,y_k^-).
\]

overall 指标为：

\[
\hat{A}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}\{\Delta_k^{(M)}>0\},
\]

\[
\hat{\mu}_M
=
\frac{1}{K}
\sum_{k=1}^{K}
\Delta_k^{(M)}.
\]

subset-level 指标为：

\[
\hat{A}_M^{(c)}
=
\frac{1}{K_c}
\sum_{k:c_k=c}
\mathbf{1}\{\Delta_k^{(M)}>0\},
\]

\[
\hat{\mu}_M^{(c)}
=
\frac{1}{K_c}
\sum_{k:c_k=c}
\Delta_k^{(M)}.
\]

## Stage 0: Server Readiness

目的：确认服务器环境、SwanLab、Hugging Face cache、Qwen 模型权重和 RewardBench 数据集都准备好。

如果需要访问 gated model，先在 Hugging Face 网页接受对应模型 license/access request，然后在服务器上执行：

```bash
huggingface-cli login
```

也可以通过环境变量提供 token：

```bash
export HF_TOKEN=hf_xxx
```

不要把 token 写入 yaml 文件。当前 cache 与 scoring 代码会读取 `HF_TOKEN`。

运行：

```bash
python scripts/verify_env.py
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
python scripts/run_experiment1.py --config configs/smoke_real_small.yaml
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
```

说明：

- `verify_env.py` 检查依赖、CUDA、GPU 和 SwanLab。
- `prepare_cache.py` 预下载配置中列出的 Hugging Face models/datasets。默认 `configs/prepare_cache.yaml` 只准备 Qwen + RewardBench 主线。
- `run_smoke_cpu.py` 验证结果文件、seed、SwanLab 记录和聚合流程。
- `smoke_real_small.yaml` 用真实 RewardBench 和 Qwen2.5-0.5B 小模型跑少量样本。
- `tune_batch_size.py` 用于决定正式实验 batch size。

## Stage 1: Qwen Family-Controlled Main Experiment

目的：在同一模型 family 中比较 base 与 instruct，检验 instruction/post-training 是否在模型 likelihood distribution 中留下可测的 ordinal trace。

设计：

- Dataset: RewardBench full。
- Models:
  - Qwen2.5-0.5B / Qwen2.5-0.5B-Instruct
  - Qwen2.5-1.5B / Qwen2.5-1.5B-Instruct
  - Qwen2.5-3B / Qwen2.5-3B-Instruct
  - Qwen2.5-7B / Qwen2.5-7B-Instruct
- Scoring: token-normalized log-likelihood。
- \(K\)：正式实验使用全量 RewardBench，不抽样。
- Prompt format: `plain`。所有 base/instruct 模型使用同一 plain prompt 拼接方式，避免 base model 无 chat template 而 instruct model 使用 chat template 带来的输入格式混淆。

运行：

```bash
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_rewardbench.yaml
```

输出：

- `metrics_overall.csv` 用于 Experiment 1。
- `metrics_by_subset.csv` 用于 Experiment 2。

解释重点：

- 如果 instruct model 的 \(\hat{A}_M\) 和 \(\hat{\mu}_M\) 通常高于 base model，说明 instruction/post-training 可能改变了模型对 reference-preferred response 的 likelihood ordering。
- 如果提升只出现在某些 subset，说明 alignment 是 distribution-dependent 的，而不是全局常数。

## Stage 1 Pilot: Stratified Sample

目的：在正式全量运行前，用较小的分层样本快速检查趋势和运行成本。

设计：

- Dataset: RewardBench stratified sample。
- Models: Qwen2.5 0.5B 和 1.5B base/instruct。
- \(K\)：由 `sample_size` 控制，当前配置为 400，并尽量按 subset 分层抽样。

运行：

```bash
python scripts/run_experiment1and2.py --config configs/pilot_qwen_rewardbench_sample.yaml
```

说明：

- Pilot 结果不作为论文主表。
- Pilot 主要用于检查趋势、字段、SwanLab 记录、batch timing 和输出目录结构。
- 主实验仍应使用 `sample_size: null` 跑全量。

## Stage 2: Preference-Specific Model Experiment (Future Extension)

目的：加入训练偏好来源更明确的模型，观察 preference/DPO training 是否更明显地改变 likelihood-induced pairwise alignment。**第一阶段不运行本实验。**

运行本阶段前，先准备扩展 cache：

```bash
python scripts/prepare_cache.py --config configs/prepare_cache_extended.yaml
```

设计：

- Dataset: RewardBench full。
- Models:
  - `mistralai/Mistral-7B-v0.1`
  - `HuggingFaceH4/zephyr-7b-beta`
- Rationale:
  - Zephyr-7B-beta 是基于 Mistral 系列的 alignment model，并公开关联 UltraChat/UltraFeedback 与 DPO-style preference alignment。
  - 这组实验比 generic Qwen instruct 更接近 preference-specific trace 的问题。
- Prompt format: `plain`。Mistral base 与 Zephyr 均使用同一 plain prompt 格式，避免 chat template 差异成为主要解释变量。

运行：

```bash
python scripts/run_experiment1and2.py --config configs/experiment1and2_zephyr_rewardbench.yaml
```

解释重点：

- 不应声称 Zephyr 使用 RewardBench 训练。
- 可以说：该实验测试一个公开 preference/DPO-aligned 模型在 broad preference benchmark 上是否表现出 stronger likelihood trace。
- 如果 Zephyr 相对 Mistral base 有更高的 `A_hat` 或 `mu_hat`，可以作为 preference optimization 改变 likelihood ordering 的补充证据。

注意：

- `mistralai/Mistral-7B-v0.1` 可能需要 Hugging Face access/token。
- 如果模型 gated 或下载失败，应先和用户确认是否替换模型。

## Stage 3: Distribution Robustness (Future Extension)

目的：检验 observable 是否依赖 reference pair distribution，而不是只在 RewardBench 上成立。**第一阶段不运行本实验。**

运行本阶段前，先准备扩展 cache：

```bash
python scripts/prepare_cache.py --config configs/prepare_cache_extended.yaml
```

设计：

- Dataset: Anthropic HH-RLHF helpful-base and harmless-base。
- Models:
  - Qwen2.5-3B / Qwen2.5-3B-Instruct
  - Qwen2.5-7B / Qwen2.5-7B-Instruct
- \(K\)：正式实验使用全量或根据资源限制后续设置 sample。
- Prompt format: `plain`。

运行：

```bash
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_hhrlhf.yaml
```

解释重点：

- 如果 RewardBench 和 HH-RLHF 上趋势一致，说明 observable 有一定跨数据集稳定性。
- 如果趋势不同，并不直接否定方法；这可能说明 likelihood-induced alignment 与 \(P_{\mathrm{pair}}\) 有关。
- HH-RLHF 的字段结构可能和 RewardBench 不完全一致，正式运行前应先做小样本 smoke。

## Recommended Order

建议执行顺序：

```bash
python scripts/verify_env.py
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
python scripts/run_experiment1.py --config configs/smoke_real_small.yaml
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
python scripts/run_experiment1and2.py --config configs/pilot_qwen_rewardbench_sample.yaml
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_rewardbench.yaml
```

后续扩展实验再执行：

```bash
python scripts/prepare_cache.py --config configs/prepare_cache_extended.yaml
python scripts/run_experiment1and2.py --config configs/experiment1and2_zephyr_rewardbench.yaml
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_hhrlhf.yaml
```

如果时间或算力有限，优先级为：

1. Stage 1 Qwen RewardBench full。
2. Stage 2 Zephyr RewardBench full。
3. Stage 3 HH-RLHF robustness。

第一阶段只需要完成第 1 项。

## Reporting

论文主表建议使用：

- `metrics_overall.csv`：报告 \(K\)、\(\hat{A}_M\)、\(\hat{m}_M^{\mathrm{sign}}\)、\(\hat{\mu}_M\)。
- `metrics_by_subset.csv`：绘制 subset-level heatmap。

其中：

\[
\hat{m}_M^{\mathrm{sign}}
=
2\hat{A}_M-1.
\]

诊断字段如 `tie_rate`、`mean_chosen_len`、`mean_rejected_len` 可以放在 appendix 或实验质量检查段落中，不作为核心 claim。
