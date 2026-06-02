from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RunConfig
from .records import PairRecord, ScoreRecord


@dataclass(frozen=True)
class BatchTiming:
    batch_index: int
    elapsed_seconds: float
    pairs: int
    tokens: int

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return math.nan
        return self.tokens / self.elapsed_seconds

    def to_json(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index,
            "elapsed_seconds": self.elapsed_seconds,
            "pairs": self.pairs,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens_per_second,
        }


class HFScorer:
    def __init__(self, model_name: str, cache_dir: Path, run: RunConfig):
        self.model_name = model_name
        self.run = run
        self.device = _resolve_device(run.device)
        dtype = _resolve_dtype(run.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir / "hub"),
            trust_remote_code=run.trust_remote_code,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir / "hub"),
            torch_dtype=dtype,
            trust_remote_code=run.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_pairs(
        self,
        pairs: list[PairRecord],
        dataset_name: str,
        scoring_rule: str,
    ) -> tuple[list[ScoreRecord], list[BatchTiming]]:
        if scoring_rule != "token_normalized_loglikelihood":
            raise ValueError(f"Unsupported scoring rule: {scoring_rule}")

        records: list[ScoreRecord] = []
        timings: list[BatchTiming] = []
        for batch_index, start in enumerate(range(0, len(pairs), self.run.batch_size)):
            batch = pairs[start : start + self.run.batch_size]
            t0 = time.perf_counter()
            chosen_scores = self._score_texts([(p.prompt, p.chosen) for p in batch])
            rejected_scores = self._score_texts([(p.prompt, p.rejected) for p in batch])
            elapsed = time.perf_counter() - t0
            token_total = sum(token_count for _, token_count in chosen_scores + rejected_scores)
            timings.append(BatchTiming(batch_index=batch_index, elapsed_seconds=elapsed, pairs=len(batch), tokens=token_total))

            for pair, chosen, rejected in zip(batch, chosen_scores, rejected_scores):
                records.append(
                    ScoreRecord(
                        pair_id=pair.pair_id,
                        subset=pair.subset,
                        model=self.model_name,
                        dataset=dataset_name,
                        score_chosen=chosen[0],
                        score_rejected=rejected[0],
                        chosen_token_count=chosen[1],
                        rejected_token_count=rejected[1],
                        scoring_rule=scoring_rule,
                    )
                )
        return records, timings

    def _score_texts(self, examples: list[tuple[str, str]]) -> list[tuple[float, int]]:
        encoded_inputs = []
        response_lengths = []
        for prompt, response in examples:
            prompt_text = self._format_prompt(prompt)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True).input_ids
            response_ids = self.tokenizer(response, add_special_tokens=False).input_ids
            full_ids = prompt_ids + response_ids
            full_ids = full_ids[-self.run.max_length :]
            response_len = min(len(response_ids), len(full_ids))
            encoded_inputs.append(full_ids)
            response_lengths.append(response_len)

        batch = self.tokenizer.pad({"input_ids": encoded_inputs}, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        results: list[tuple[float, int]] = []
        for row_idx, response_len in enumerate(response_lengths):
            seq_len = int(attention_mask[row_idx].sum().item())
            end = max(seq_len - 1, 0)
            start = max(end - response_len, 0)
            selected = token_log_probs[row_idx, start:end]
            count = int(selected.numel())
            score = float(selected.mean().item()) if count else float("nan")
            results.append((score, count))
        return results

    def _format_prompt(self, prompt: str) -> str:
        if self.run.prompt_format == "plain":
            return prompt.rstrip() + "\n"
        if self.run.prompt_format == "chat":
            if not (hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template):
                raise ValueError(f"Model {self.model_name} does not define a chat template.")
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.run.prompt_format == "native":
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt.rstrip() + "\n"
        raise ValueError(f"Unsupported prompt_format: {self.run.prompt_format}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
