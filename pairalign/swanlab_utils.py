from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


class SwanLabRun:
    def __init__(self, enabled: bool, project: str, workspace: str | None, name: str, config: Any):
        self.enabled = enabled
        self._swanlab = None
        self.init_error: str | None = None
        if not enabled:
            return
        try:
            import swanlab

            self._swanlab = swanlab
            init_kwargs = {
                "project": project,
                "experiment_name": name,
                "config": _to_plain(config),
            }
            if workspace:
                init_kwargs["workspace"] = workspace
            swanlab.init(**init_kwargs)
        except Exception as exc:
            self.init_error = str(exc)
            print(f"[swanlab] disabled after init failure: {exc}")
            self.enabled = False

    @property
    def init_ok(self) -> bool:
        return self.enabled and self._swanlab is not None and self.init_error is None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._swanlab is None:
            return
        try:
            if step is None:
                self._swanlab.log(data)
            else:
                self._swanlab.log(data, step=step)
        except Exception as exc:
            print(f"[swanlab] log failed: {exc}")

    def finish(self) -> None:
        if self.enabled and self._swanlab is not None:
            try:
                self._swanlab.finish()
            except Exception as exc:
                print(f"[swanlab] finish failed: {exc}")


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "__fspath__"):
        return str(value)
    return value
