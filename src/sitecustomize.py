from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    value = os.getenv("MM_SIDECAR_AUTO_PATCH", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


if _enabled():
    try:
        from mm_sidecar.integrations.vllm_patch.patches import apply_monkey_patches

        apply_monkey_patches()
    except Exception as exc:
        message = (
            "mm-sidecar sitecustomize auto patch failed: "
            f"{exc.__class__.__name__}: {exc}\n"
        )
        sys.stderr.write(message)
        strict = os.getenv("MM_SIDECAR_AUTO_PATCH_STRICT", "1").strip().lower()
        if strict in {"1", "true", "yes", "on"}:
            raise
