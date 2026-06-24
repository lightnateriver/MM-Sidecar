from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from mm_sidecar.contracts import ArtifactDescriptor

from .config import MemoryCacheConfig


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass(slots=True)
class _ReusableEntry:
    descriptor: ArtifactDescriptor
    payload: Any
    created_at_ms: float
    last_accessed_at_ms: float
    expires_at_ms: float

    @property
    def size_bytes(self) -> int:
        payload = self.payload
        if hasattr(payload, "nbytes"):
            return int(payload.nbytes)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return len(payload)
        return 0


class CpuMemoryCachePool:
    def __init__(self, config: MemoryCacheConfig | None = None) -> None:
        self._config = config or MemoryCacheConfig()
        self._reusable: "OrderedDict[str, _ReusableEntry]" = OrderedDict()
        self._inflight: set[str] = set()
        self._reusable_bytes = 0

    def mark_inflight(self, cache_key: str) -> None:
        self._inflight.add(cache_key)

    def clear_inflight(self, cache_key: str) -> None:
        self._inflight.discard(cache_key)

    def is_inflight(self, cache_key: str) -> bool:
        return cache_key in self._inflight

    def get(self, cache_key: str) -> tuple[ArtifactDescriptor, Any] | None:
        self._evict_expired()
        entry = self._reusable.get(cache_key)
        if entry is None:
            return None
        entry.last_accessed_at_ms = _now_ms()
        self._reusable.move_to_end(cache_key)
        return entry.descriptor, entry.payload

    def put(self, cache_key: str, descriptor: ArtifactDescriptor, payload: Any) -> None:
        now_ms = _now_ms()
        expires_at_ms = now_ms + self._config.reusable_entry_ttl_s * 1000.0
        previous = self._reusable.pop(cache_key, None)
        if previous is not None:
            self._reusable_bytes -= previous.size_bytes
        entry = _ReusableEntry(
            descriptor=descriptor,
            payload=payload,
            created_at_ms=now_ms,
            last_accessed_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
        )
        self._reusable[cache_key] = entry
        self._reusable_bytes += entry.size_bytes
        self._evict_expired()
        self._evict_lru_until_within_budget()

    def has_reusable(self, cache_key: str) -> bool:
        return self.get(cache_key) is not None

    def stats(self) -> dict[str, int]:
        self._evict_expired()
        return {
            "reusable_items": len(self._reusable),
            "reusable_bytes": self._reusable_bytes,
            "inflight_items": len(self._inflight),
        }

    def _evict_expired(self) -> None:
        now_ms = _now_ms()
        expired_keys = [
            cache_key
            for cache_key, entry in self._reusable.items()
            if entry.expires_at_ms <= now_ms
        ]
        for cache_key in expired_keys:
            entry = self._reusable.pop(cache_key)
            self._reusable_bytes -= entry.size_bytes

    def _evict_lru_until_within_budget(self) -> None:
        max_bytes = self._config.max_reusable_bytes
        while self._reusable and self._reusable_bytes > max_bytes:
            _, entry = self._reusable.popitem(last=False)
            self._reusable_bytes -= entry.size_bytes
