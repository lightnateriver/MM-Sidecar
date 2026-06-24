"""Monkey-patch integration for pip-installed vLLM runtimes.

The package stays importable without ``vllm`` installed so local unit tests can
exercise sidecar-only logic. The actual vLLM dependency is resolved lazily when
the monkey patch entrypoints are called.
"""

from __future__ import annotations

from typing import Any

__all__ = ["apply_monkey_patches", "get_patch_state"]


def apply_monkey_patches() -> None:
    from .patches import apply_monkey_patches as _apply_monkey_patches

    _apply_monkey_patches()


def get_patch_state() -> dict[str, Any]:
    from .patches import get_patch_state as _get_patch_state

    return _get_patch_state()
