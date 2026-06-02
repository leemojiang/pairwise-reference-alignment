
进入AutoDL服务器之后:

## 依赖管理
有UV情况
``` 
uv sync --extra tracking
```

没有UV情况
```
pip install -r requirements.txt
```

## 登录swanlab (用户操作,不需要自动执行)

```
swanlab login
```

## 登录 HuggingFace(用户操作,不需要自动执行)

```
huggingface-cli login
```
或者设置环境变量：
```
export HF_TOKEN=hf_xxx
```

## 检查环境以及数据集是否准备好
```
python scripts/verify_env.py 
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
```

## Smoke Test

```
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
python scripts/run_experiment1.py --config configs/smoke_real_small.yaml


```


## (可选)检查是否满足实验参数,实验初步迭代
```
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
```


## 执行实验
```
python scripts/run_experiment1.py --config configs/experiment1.yaml
python scripts/run_experiment2.py --config configs/experiment2.yaml
```

## 在服务器上建议运行顺序
主要服务器运行顺序：

python scripts/verify_env.py
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
python scripts/run_experiment1.py --config configs/smoke_real_small.yaml
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
python scripts/run_experiment1.py --config configs/experiment1.yaml
python scripts/run_experiment2.py --config configs/experiment2.yaml

## Agent 执行建议

本节用于指导 agent 进入服务器后按固定流程检查环境、准备缓存、运行 smoke test 和正式实验。除非用户明确要求跳过某一步，否则不要直接从正式实验开始。

### 0. 基本原则

- 每一步先看终端输出，再决定是否继续下一步。
- 如果某一步失败，优先定位失败原因，不要继续跑正式实验。
- 不要删除已有 `runs/` 结果目录；如果需要重新运行，使用新的 run 目录保存结果。
- 不要手动修改已经生成的 `raw_scores.jsonl`、`metrics_overall.csv`、`metrics_by_subset.csv`。
- 服务器环境通常不使用 uv 管理，因此默认使用 `pip install -r requirements.txt`。
- 如果服务器已经预装 torch / transformers，仍然需要确认 `datasets`、`huggingface-hub`、`pyyaml`、`numpy`、`swanlab` 是否可导入。
- SwanLab 登录通常需要用户操作。agent 可以运行检查脚本，但不要伪造登录状态。

### 1. 依赖安装

如果服务器没有 uv，执行：

```bash
pip install -r requirements.txt
```

如果用户明确说明服务器使用 uv，才执行：

```bash
uv sync --extra tracking
```

安装后必须执行环境检查：

```bash
python scripts/verify_env.py
```

通过标准：

- `torch` 可以导入；
- `transformers`、`datasets`、`huggingface_hub`、`numpy`、`yaml`、`swanlab` 可以导入；
- `cuda_available` 为 `true`；
- `cuda_device_count` 大于 0；
- `cuda_devices` 中能看到显卡名称和显存；
- `swanlab_init_ok` 为 `true`。如果为 `false`，需要让用户执行 `swanlab login` 后再重试。

### 2. 准备 Hugging Face 缓存

在正式实验前，先统一下载实验需要的数据集和模型：

```bash
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
```

默认 cache 配置只准备 Qwen + RewardBench 主线。扩展实验需要额外执行：

```bash
python scripts/prepare_cache.py --config configs/prepare_cache_extended.yaml
```

通过标准：

- `cache_env.json` 中的 `cache_dir` 指向服务器大容量磁盘；
- `hf_endpoint` 正确，例如 AutoDL 可使用 `https://hf-mirror.com`；
- 每个 `dataset` 项 `ok=true`；
- 每个 `model` 项 `ok=true`；
- `cache_items.jsonl` 中没有失败项。

如果模型下载失败：

- 先检查网络和 `HF_ENDPOINT`；
- 再检查 Hugging Face token；
- 如果是 gated model，不要强行换模型，应先和用户确认。
- 主线实验优先使用 `configs/prepare_cache.yaml`；Zephyr/Mistral 和 HH-RLHF 扩展实验使用 `configs/prepare_cache_extended.yaml`。

### 3. CPU Smoke Test

CPU smoke 只验证实验流程可追溯，不验证真实模型 likelihood 是否正确：

```bash
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
```

通过标准：

- 生成新的 `runs/*_smoke_cpu/` 目录；
- 目录中存在 `config.yaml`、`raw_scores.jsonl`、`metrics_overall.csv`、`metrics_by_subset.csv`、`run_meta.json`；
- `run_meta.json` 中保存了 `seed`、`cache_dir`、`pair_count`、`swanlab_init_ok`。

### 4. 真实小样本 Smoke Test

在正式跑全量实验前，用真实 Hugging Face 数据集和小模型跑少量样本：

```bash
python scripts/run_experiment1.py --config configs/smoke_real_small.yaml
```

通过标准：

- 能成功加载 RewardBench；
- 能成功加载 Qwen2.5-0.5B base / instruct；
- 能生成 `raw_scores.jsonl` 和 `metrics_overall.csv`；
- `batch_timings.jsonl` 中有每个 batch 的 `elapsed_seconds` 和 `tokens_per_second`；
- 没有 CUDA OOM。

### 5. Batch Size 初测

用小样本配置测试当前显卡可承受的 batch size：

```bash
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
```

判断原则：

- 优先选择不会 OOM 的最大稳定 batch size；
- 30G 显存下，7B 模型建议先使用 batch size 2；
- 如果 batch size 4 在 0.5B 上可行，不代表 7B 上一定可行；
- 正式实验前不要只看速度，也要确认结果文件正常生成。

### 6. 正式实验

实验 1：overall likelihood-induced pairwise alignment。

```bash
python scripts/run_experiment1.py --config configs/experiment1.yaml
```

实验 2：subset-level alignment map。

```bash
python scripts/run_experiment2.py --config configs/experiment2.yaml
```

正式实验通过标准：

- 每个实验生成独立 run 目录；
- `raw_scores.jsonl` 中每个模型、数据集、样本都有记录；
- 实验 1 至少生成 `metrics_overall.csv`；
- 实验 2 额外生成 `metrics_by_subset.csv`；
- `run_meta.json` 保存模型列表、数据集列表、scoring rule、batch size、max length、dtype、device、seed；
- SwanLab 中可以看到 batch timing 和最终 metrics。

### 7. 失败处理建议

- 依赖导入失败：先安装或升级 `requirements.txt` 中的包，再重跑 `verify_env.py`。
- SwanLab 失败：让用户执行 `swanlab login`，然后重跑 `verify_env.py`。
- Hugging Face 下载失败：检查 `HF_ENDPOINT`、token、磁盘空间和模型是否 gated。
- CUDA OOM：先把 batch size 降到 1；如果仍失败，降低 `max_length` 或先跳过 7B。
- 某个模型失败：不要删除已完成模型的结果；记录失败模型和错误，继续前先和用户确认。
- 指标异常：先查看 `raw_scores.jsonl`、`batch_timings.jsonl` 和结果解析文档，不要直接重跑覆盖。

## 第一阶段执行范围

第一阶段只运行 Qwen + RewardBench 主线，不运行 Zephyr/Mistral 和 HH-RLHF 扩展实验。

Hugging Face token 建议通过服务器登录或环境变量提供：

```bash
huggingface-cli login
```

或者：

```bash
export HF_TOKEN=hf_xxx
```

不要把 token 写入 yaml 文件。当前 `prepare_cache.py` 和模型加载代码会读取 `HF_TOKEN`。

第一阶段推荐命令：

```bash
python scripts/verify_env.py
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
python scripts/run_experiment1.py --config configs/smoke_real_small.yaml
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
python scripts/run_experiment1and2.py --config configs/pilot_qwen_rewardbench_sample.yaml
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_rewardbench.yaml
```

后续扩展实验才使用：

```bash
python scripts/prepare_cache.py --config configs/prepare_cache_extended.yaml
python scripts/run_experiment1and2.py --config configs/experiment1and2_zephyr_rewardbench.yaml
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_hhrlhf.yaml
```
