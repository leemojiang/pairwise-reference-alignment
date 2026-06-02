# Pairwise Reference Alignment

This repository contains the experiment code for **Pairwise Reference Alignment as a Model-Level Ordinal Observable**.

Paper: [Pairwise Reference Alignment as a Model-Level Ordinal Observable](https://arxiv.org/abs/2605.30758)

The project studies whether a fixed language model's score-induced ordering agrees with a reference pair distribution. Given a prompt $x$, a reference-preferred response $y^+$, a reference-rejected response $y^-$, and a model score $S_M(x,y)$, the code reports two main observables.

The sign agreement observable is

$$
\hat{A}_M=\frac{1}{K}\sum_{k=1}^{K}\mathbf{1}\left[S_M(x_k,y_k^+) > S_M(x_k,y_k^-)\right].
$$

The mean signed margin observable is

$$
\hat{\mu}_M=\frac{1}{K}\sum_{k=1}^{K}\left[S_M(x_k,y_k^+) - S_M(x_k,y_k^-)\right].
$$

The current experiments mainly use token-normalized log-likelihood as the scoring rule and evaluate Qwen2.5 models on RewardBench.

## Citation

```bibtex
@misc{li2026pairwisereferencealignmentmodellevel,
      title={Pairwise Reference Alignment as a Model-Level Ordinal Observable},
      author={Mujing Li},
      year={2026},
      eprint={2605.30758},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.30758},
}
```

## Repository Structure

```text
.
├── configs/                 # YAML configs for cache preparation, smoke tests, and experiments
├── docs/                    # Usage notes, result-file explanations, and experiment reports
├── fixtures/                # Small toy preference-pair data for local smoke tests
├── pairalign/               # Core library code
├── plot/                    # Plotting scripts for experiment figures
├── scripts/                 # Command-line entry points for verification and experiments
├── tests/                   # Unit tests for metrics, records, and finite-sample utilities
├── main.py                  # Unified helper CLI
├── pyproject.toml           # uv / Python project metadata
├── requirements.txt         # pip install fallback for server environments
└── README.md
```

Generated or machine-local directories may also appear during development:

```text
cache/                       # Hugging Face model/dataset cache, if configured locally
runs/                        # Local experiment outputs
remote_results/              # Downloaded remote experiment outputs
swanlog/                     # SwanLab logs
.venv/, .uv-cache/           # Local uv/Python environment files
```

These generated directories are not required to understand or run the code and should usually not be included in a public release.

## Installation

This project was developed with Python 3.12.

With `uv`:

```bash
uv sync --extra tracking
```

Without `uv`, for example on a GPU server:

```bash
pip install -r requirements.txt
```

The `tracking` extra installs SwanLab support. If you do not want experiment tracking, pass `--no-swanlab` where supported.

## Environment and Cache Preparation

Check that the Python packages, CUDA runtime, GPU visibility, and SwanLab logging are available:

```bash
python scripts/verify_env.py
```

Prepare the Hugging Face model and dataset cache for the main Qwen + RewardBench experiments:

```bash
python scripts/prepare_cache.py --config configs/prepare_cache.yaml
```

For Hugging Face authentication, login manually or provide a token through the environment:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=hf_xxx
```

Do not write Hugging Face tokens into config files.

## Running Smoke Tests

Run the CPU smoke test first. This checks the local experiment flow, output structure, and result tracking without requiring a GPU:

```bash
python scripts/run_smoke_cpu.py --config configs/smoke_cpu.yaml
```

Then run a small real-data smoke test before launching the full experiment:

```bash
python scripts/run_experiment1and2.py --config configs/pilot_qwen_rewardbench_sample.yaml
```

Optional batch-size tuning:

```bash
python scripts/tune_batch_size.py --config configs/smoke_real_small.yaml --batch-sizes 1,2,4
```

## Main Experiments

Experiment 1 and Experiment 2 are merged into one entry point. The same run produces both overall metrics and subset-level metrics.

Run the main Qwen + RewardBench experiment:

```bash
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_rewardbench.yaml
```

Run the chat-template ablation:

```bash
python scripts/run_experiment1and2.py --config configs/experiment1and2_qwen_rewardbench_chat.yaml
```

Run finite-sample subset analysis:

```bash
python scripts/run_experiment4.py --config configs/experiment4_qwen_rewardbench_finite_sample.yaml
```

Run bootstrap confidence-interval analysis:

```bash
python scripts/run_experiment4b.py --config configs/experiment4b_qwen_rewardbench_bootstrap.yaml
```

The unified CLI in `main.py` exposes the same basic workflow:

```bash
python main.py verify-env
python main.py prepare-cache --config configs/prepare_cache.yaml
python main.py smoke-cpu --config configs/smoke_cpu.yaml
python main.py experiment1and2 --config configs/experiment1and2_qwen_rewardbench.yaml
```

## Outputs

Each experiment creates a timestamped run directory under `runs/`, for example:

```text
runs/20260528-110013_experiment4b_qwen_rewardbench_bootstrap/
```

Common output files include:

- `config.yaml`: a copy of the config used for the run.
- `run_meta.json`: model, dataset, seed, device, dtype, batch size, and other run metadata.
- `raw_scores.jsonl`: per-pair raw scores and margins.
- `batch_timings.jsonl`: batch-level runtime and throughput records.
- `metrics_overall.csv`: overall alignment and margin metrics.
- `metrics_by_subset.csv`: subset-level alignment and margin metrics.

Finite-sample and bootstrap experiments produce additional CSV files for subset trials and confidence intervals. See `docs/ResultFiles.md`, `docs/Experiment4FiniteSample.md`, and `docs/Experiment4Bootstrap.md` for detailed field definitions.

## Plotting

Plotting scripts are in `plot/`. They expect experiment result CSV files to be available locally and write figures to the configured output directory.

On Windows:

```bat
plot\run_all_plots.bat
```

Individual matplotlib scripts can also be run directly, for example:

```bash
python plot/plot_overall_bars_mpl.py
python plot/plot_subset_family_radar_mpl.py
python plot/plot_finite_sample_representative_mpl.py
```

See `plot/README.md` for plot-specific path configuration.

## Notes for Reproduction

- The first-stage experiments focus on Qwen2.5 models and RewardBench.
- `configs/prepare_cache.yaml` prepares the main experiment cache.
- `configs/prepare_cache_extended.yaml` is reserved for extended model/dataset experiments.
- Use a GPU server for full likelihood scoring. Local CPU tests are intended only for flow validation.
- Keep `raw_scores.jsonl` when possible: it is the most useful intermediate file for later analysis.
- SwanLab logging is optional but useful for tracking environment checks, cache preparation, batch timing, and final metrics.

## Opportunity Note

I am currently looking for RA, research internship, or full-time opportunities related to LLM alignment, evaluation, statistical modeling, and AI research engineering. If this project is relevant to your group or team, feel free to contact me.
