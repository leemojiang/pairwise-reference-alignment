from __future__ import annotations

from pathlib import Path
from typing import Any

from .records import PairRecord


def load_pair_dataset(dataset_spec: str | dict[str, Any], cache_dir: Path, limit: int | None = None) -> list[PairRecord]:
    spec = _normalize_dataset_spec(dataset_spec)
    dataset_id = _dataset_id(spec)
    name = spec["name"]
    if name.endswith(".jsonl") or Path(name).exists():
        records = _load_jsonl_pairs(Path(name), limit)
        if records:
            return records

    from datasets import DatasetDict, load_dataset

    load_kwargs = {"cache_dir": str(cache_dir / "datasets")}
    if spec.get("subset") is not None:
        loaded = load_dataset(name, spec["subset"], **load_kwargs)
    else:
        loaded = load_dataset(name, **load_kwargs)
    if isinstance(loaded, DatasetDict):
        split_name = spec.get("split") or ("train" if "train" in loaded else next(iter(loaded.keys())))
        dataset = loaded[split_name]
    else:
        dataset = loaded
    records: list[PairRecord] = []
    for idx, row in enumerate(dataset):
        pair = normalize_pair(row, idx, dataset_id=dataset_id)
        if pair is not None:
            records.append(pair)
        if limit is not None and len(records) >= limit:
            break
    return records


def dataset_label(dataset_spec: str | dict[str, Any]) -> str:
    return _dataset_id(_normalize_dataset_spec(dataset_spec))


def normalize_pair(row: dict[str, Any], idx: int, dataset_id: str | None = None) -> PairRecord | None:
    prompt = _first_text(row, ["prompt", "instruction", "question", "input"])
    chosen = _first_text(row, ["chosen", "response_chosen", "winner", "accepted"])
    rejected = _first_text(row, ["rejected", "response_rejected", "loser", "discarded"])

    if prompt is None and isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
        parsed = _parse_shared_transcript(row["chosen"], row["rejected"])
        if parsed is not None:
            prompt, chosen, rejected = parsed
    elif prompt is None and isinstance(row.get("chosen"), list):
        chosen_messages = row.get("chosen") or []
        rejected_messages = row.get("rejected") or []
        prompt = _messages_to_prompt(chosen_messages)
        chosen = _messages_to_last_assistant(chosen_messages)
        rejected = _messages_to_last_assistant(rejected_messages)

    if prompt is None or chosen is None or rejected is None:
        return None

    pair_id = str(row.get("id") or row.get("pair_id") or row.get("example_id") or f"pair_{idx:06d}")
    subset = str(row.get("subset") or row.get("category") or row.get("domain") or row.get("source") or dataset_id or "overall")
    return PairRecord(pair_id=pair_id, prompt=prompt, chosen=chosen, rejected=rejected, subset=subset)


def _normalize_dataset_spec(spec: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"name": spec}
    return dict(spec)


def _dataset_id(spec: dict[str, Any]) -> str:
    parts = [str(spec["name"])]
    if spec.get("subset"):
        parts.append(str(spec["subset"]))
    if spec.get("split"):
        parts.append(str(spec["split"]))
    return ":".join(parts)


def _first_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str | None:
    user_parts = [str(m.get("content", "")) for m in messages if m.get("role") in {"user", "human"}]
    return "\n".join(part for part in user_parts if part).strip() or None


def _messages_to_last_assistant(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") in {"assistant", "gpt"}:
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return None


def _parse_shared_transcript(chosen_text: str, rejected_text: str) -> tuple[str, str, str] | None:
    chosen_parts = chosen_text.rsplit("\n\nAssistant:", 1)
    rejected_parts = rejected_text.rsplit("\n\nAssistant:", 1)
    if len(chosen_parts) != 2 or len(rejected_parts) != 2:
        return None

    chosen_prompt, chosen_response = chosen_parts[0].strip(), chosen_parts[1].strip()
    rejected_prompt, rejected_response = rejected_parts[0].strip(), rejected_parts[1].strip()
    prompt = _longest_common_prefix(chosen_prompt, rejected_prompt).strip()
    if not prompt:
        prompt = chosen_prompt
    if not chosen_response or not rejected_response:
        return None
    return prompt, chosen_response, rejected_response


def _longest_common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return left[:idx]


def _load_jsonl_pairs(path: Path, limit: int | None) -> list[PairRecord]:
    import json

    records: list[PairRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            pair = normalize_pair(json.loads(line), idx)
            if pair is not None:
                records.append(pair)
            if limit is not None and len(records) >= limit:
                break
    return records
