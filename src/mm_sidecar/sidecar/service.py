from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any

from .config import SidecarManagerConfig, WorkerPoolConfig
from .manager import SidecarManager
from .processor import InlineProcessorWorkerPool, MultiProcessProcessorWorkerPool

SidecarAddress = str | tuple[str, int]


def _safe_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _available_cpu_ids() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
        if cpu_ids:
            return cpu_ids
    cpu_count = os.cpu_count() or 1
    return tuple(range(cpu_count))


def _parse_cpu_set(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    cpu_ids: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            cpu_ids.extend(range(start, end + 1))
            continue
        try:
            cpu_ids.append(int(token))
        except ValueError:
            continue
    unique_ids = tuple(sorted(set(cpu_ids)))
    return unique_ids or None


def _default_worker_count() -> int:
    return max(1, min(32, len(_available_cpu_ids())))


def _default_cpu_affinity_map(
    worker_count: int,
    cpu_ids: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], ...]:
    cpu_ids = cpu_ids or _available_cpu_ids()
    return tuple((cpu_ids[index % len(cpu_ids)],) for index in range(worker_count))


def _manager_config_from_env(start_method: str) -> SidecarManagerConfig:
    worker_count = _safe_int(
        os.getenv("MM_SIDECAR_WORKER_COUNT"),
        _default_worker_count(),
    )
    base_config = SidecarManagerConfig()
    cache_config = base_config.cache.__class__(
        max_reusable_bytes=_safe_int(
            os.getenv("MM_SIDECAR_REUSABLE_CACHE_BYTES"),
            base_config.cache.max_reusable_bytes,
        ),
        reusable_entry_ttl_s=_safe_float(
            os.getenv("MM_SIDECAR_REUSABLE_TTL_S"),
            base_config.cache.reusable_entry_ttl_s,
        ),
    )
    worker_cpu_ids = _parse_cpu_set(os.getenv("MM_SIDECAR_WORKER_CPU_SET"))
    if worker_cpu_ids is not None:
        cpu_affinity_map = _default_cpu_affinity_map(worker_count, worker_cpu_ids)
    else:
        cpu_affinity_map = _default_cpu_affinity_map(worker_count)
    return SidecarManagerConfig(
        cache=cache_config,
        workers=WorkerPoolConfig(
            worker_count=worker_count,
            cpu_affinity_map=cpu_affinity_map,
            start_method=start_method,
        ),
    )


def _bind_current_process_cpu(cpu_ids: tuple[int, ...] | None) -> None:
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, set(cpu_ids))
    except OSError:
        return


def _build_worker_pool(config: SidecarManagerConfig, mode: str):
    normalized_mode = mode.strip().lower()
    if normalized_mode == "inline":
        return InlineProcessorWorkerPool(
            worker_count=config.workers.worker_count
        )
    return MultiProcessProcessorWorkerPool(config.workers)


def _normalize_socket_path(socket_path: str | None) -> str:
    if socket_path:
        return socket_path
    temp_dir = Path(tempfile.gettempdir()) / "mm-sidecar"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return str(temp_dir / f"sidecar-{uuid.uuid4().hex}.sock")


def _cleanup_socket(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _normalize_transport(transport: str) -> str:
    normalized = transport.strip().lower()
    if normalized in {"auto", "unix", "tcp"}:
        return normalized
    raise ValueError(f"unsupported sidecar transport: {transport}")


def _listener_candidates(
    transport: str,
    socket_path: str | None,
    tcp_host: str,
    tcp_port: int,
) -> Sequence[tuple[str, SidecarAddress]]:
    normalized_transport = _normalize_transport(transport)
    candidates: list[tuple[str, SidecarAddress]] = []
    if normalized_transport in {"auto", "unix"}:
        candidates.append(("AF_UNIX", _normalize_socket_path(socket_path)))
    if normalized_transport in {"auto", "tcp"}:
        candidates.append(("AF_INET", (tcp_host, tcp_port)))
    return tuple(candidates)


def _create_listener(
    *,
    family: str,
    address: SidecarAddress,
) -> Listener:
    if family == "AF_UNIX":
        assert isinstance(address, str)
        _cleanup_socket(address)
    return Listener(address=address, family=family)


def _serve_forever(
    transport: str,
    socket_path: str | None,
    tcp_host: str,
    tcp_port: int,
    ready_queue: Any,
    manager_config: SidecarManagerConfig,
    worker_pool_mode: str,
    control_cpu_affinity: tuple[int, ...] | None,
) -> None:
    manager = SidecarManager(
        config=manager_config,
        worker_pool=_build_worker_pool(manager_config, worker_pool_mode),
    )
    listener: Listener | None = None
    bound_family = ""
    bound_address: SidecarAddress | None = None
    errors: list[str] = []
    try:
        for candidate_family, candidate_address in _listener_candidates(
            transport=transport,
            socket_path=socket_path,
            tcp_host=tcp_host,
            tcp_port=tcp_port,
        ):
            try:
                listener = _create_listener(
                    family=candidate_family,
                    address=candidate_address,
                )
                bound_family = candidate_family
                bound_address = listener.address
                # Keep startup behavior close to the earlier validated runs:
                # build the worker pool and listener first, then narrow the
                # control-plane affinity for the long-running manager loop.
                _bind_current_process_cpu(control_cpu_affinity)
                ready_queue.put(
                    {
                        "ok": True,
                        "family": bound_family,
                        "address": bound_address,
                    }
                )
                break
            except Exception as exc:  # pragma: no cover - env-specific failure path
                errors.append(
                    f"{candidate_family}({candidate_address!r}): "
                    f"{exc.__class__.__name__}: {exc}"
                )
        if listener is None:
            ready_queue.put(
                {
                    "ok": False,
                    "error": "; ".join(errors)
                    or "no sidecar listener candidates available",
                }
            )
            return
    except Exception as exc:  # pragma: no cover - child bootstrap failure
        ready_queue.put(
            {
                "ok": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        return

    try:
        while True:
            conn = listener.accept()
            try:
                method_name, args, kwargs = conn.recv()
                if method_name == "shutdown":
                    conn.send({"ok": True})
                    break
                method = getattr(manager, method_name)
                result = method(*args, **kwargs)
                conn.send({"ok": True, "result": result})
            except Exception as exc:
                conn.send(
                    {
                        "ok": False,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
            finally:
                conn.close()
    finally:
        if listener is not None:
            listener.close()
        manager.close()
        if bound_family == "AF_UNIX" and isinstance(bound_address, str):
            _cleanup_socket(bound_address)


@dataclass(frozen=True, slots=True)
class SidecarServiceConfig:
    transport: str = "auto"
    socket_path: str | None = None
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 0
    manager: SidecarManagerConfig = SidecarManagerConfig()
    worker_pool_mode: str = "process"
    start_method: str = "fork"
    control_cpu_affinity: tuple[int, ...] | None = None


class SidecarClient:
    def __init__(
        self,
        address: SidecarAddress,
        *,
        family: str = "AF_UNIX",
    ) -> None:
        self.address = address
        self.family = family
        self.socket_path = address if family == "AF_UNIX" else None

    def _request(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        conn = Client(address=self.address, family=self.family)
        try:
            conn.send((method_name, args, kwargs))
            payload = conn.recv()
        finally:
            conn.close()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "sidecar request failed"))
        return payload.get("result")

    def prepare(self, descriptors: list[Any] | tuple[Any, ...]):
        return self._request("prepare", descriptors)

    def batch_get_status(self, handles: list[Any] | tuple[Any, ...]):
        return self._request("batch_get_status", handles)

    def lookup_by_cache_keys(self, cache_keys: list[str] | tuple[str, ...]):
        return self._request("lookup_by_cache_keys", cache_keys)

    def wait_for_states(
        self,
        handles: list[Any] | tuple[Any, ...],
        target_states: set[Any],
        timeout_ms: float,
        poll_interval_ms: float = 1.0,
    ):
        return self._request(
            "wait_for_states",
            handles,
            target_states,
            timeout_ms,
            poll_interval_ms,
        )

    def wait_for_metadata(
        self,
        handles: list[Any] | tuple[Any, ...],
        timeout_ms: float,
        poll_interval_ms: float = 1.0,
    ):
        return self._request(
            "wait_for_metadata",
            handles,
            timeout_ms,
            poll_interval_ms,
        )

    def fetch_ready(self, handle: Any):
        return self._request("fetch_ready", handle)

    def try_fallback_claim(
        self,
        handles: list[Any] | tuple[Any, ...],
        claimer_id: str,
    ):
        return self._request("try_fallback_claim", handles, claimer_id)

    def mark_fallback_local_done(self, handle: Any, claimer_id: str):
        return self._request("mark_fallback_local_done", handle, claimer_id)

    def stats(self):
        return self._request("stats")

    def shutdown(self) -> None:
        self._request("shutdown")

    def close(self) -> None:
        return None


class SidecarServiceProcess:
    def __init__(self, config: SidecarServiceConfig | None = None) -> None:
        self.config = config or SidecarServiceConfig()
        self.socket_path = (
            _normalize_socket_path(self.config.socket_path)
            if _normalize_transport(self.config.transport) in {"auto", "unix"}
            else None
        )
        self._family: str | None = None
        self._address: SidecarAddress | None = None
        ctx = mp.get_context(self.config.start_method)
        self._ready_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_serve_forever,
            args=(
                self.config.transport,
                self.socket_path,
                self.config.tcp_host,
                self.config.tcp_port,
                self._ready_queue,
                self.config.manager,
                self.config.worker_pool_mode,
                self.config.control_cpu_affinity,
            ),
        )
        self._started = False

    def start(self) -> SidecarClient:
        if not self._started:
            self._process.start()
            self._started = True
            try:
                payload = self._ready_queue.get(timeout=5.0)
            except Exception as exc:
                raise TimeoutError("Timed out waiting for sidecar readiness") from exc
            if not payload.get("ok"):
                raise RuntimeError(
                    payload.get("error", "sidecar process failed to start")
                )
            self._family = str(payload["family"])
            self._address = payload["address"]
            if self._family == "AF_UNIX" and isinstance(self._address, str):
                self.socket_path = self._address
        assert self._address is not None
        assert self._family is not None
        return SidecarClient(self._address, family=self._family)

    def join(self, timeout: float | None = None) -> None:
        if self._started:
            self._process.join(timeout=timeout)

    def terminate(self) -> None:
        if self._started and self._process.is_alive():
            if self._address is not None and self._family is not None:
                try:
                    SidecarClient(self._address, family=self._family).shutdown()
                except Exception:
                    pass
                self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=2.0)
        if self._family == "AF_UNIX" and isinstance(self._address, str):
            _cleanup_socket(self._address)


def create_sidecar_client(
    address: SidecarAddress,
    *,
    family: str = "AF_UNIX",
) -> SidecarClient:
    return SidecarClient(address, family=family)


def sidecar_service_config_from_env(
    *,
    required: bool = True,
) -> SidecarServiceConfig | None:
    transport = os.getenv("MM_SIDECAR_TRANSPORT", "auto").strip().lower() or "auto"
    socket_path = os.getenv("MM_SIDECAR_SOCKET_PATH") or None
    tcp_host = os.getenv("MM_SIDECAR_TCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_tcp_port = os.getenv("MM_SIDECAR_TCP_PORT")
    try:
        tcp_port = int(raw_tcp_port) if raw_tcp_port is not None else 0
    except ValueError:
        tcp_port = 0
    worker_pool_mode = (
        os.getenv("MM_SIDECAR_WORKER_POOL_MODE", "process").strip().lower()
        or "process"
    )
    start_method = (
        os.getenv("MM_SIDECAR_WORKER_START_METHOD", "fork").strip().lower() or "fork"
    )
    manager_config = _manager_config_from_env(start_method)
    control_cpu_affinity = _parse_cpu_set(os.getenv("MM_SIDECAR_CONTROL_CPU_SET"))
    resolved_transport = transport
    if resolved_transport == "auto":
        if socket_path:
            resolved_transport = "unix"
        elif tcp_port > 0:
            resolved_transport = "tcp"
        else:
            if not required:
                return None
            raise RuntimeError(
                "MM_SIDECAR_SOCKET_PATH or MM_SIDECAR_TCP_PORT must be set "
                "for an independently launched sidecar service"
            )
    if resolved_transport == "unix" and not socket_path:
        raise RuntimeError(
            "MM_SIDECAR_SOCKET_PATH must be set when MM_SIDECAR_TRANSPORT=unix"
        )
    if resolved_transport == "tcp" and tcp_port <= 0:
        raise RuntimeError(
            "MM_SIDECAR_TCP_PORT must be set when MM_SIDECAR_TRANSPORT=tcp"
        )
    return SidecarServiceConfig(
        transport=resolved_transport,
        socket_path=socket_path,
        tcp_host=tcp_host,
        tcp_port=tcp_port,
        manager=manager_config,
        worker_pool_mode=worker_pool_mode,
        start_method=start_method,
        control_cpu_affinity=control_cpu_affinity,
    )


def connect_sidecar_client_from_env(
    *,
    required: bool = False,
) -> SidecarClient | None:
    config = sidecar_service_config_from_env(required=required)
    if config is None:
        return None
    transport = _normalize_transport(config.transport)
    if transport == "tcp":
        return create_sidecar_client(
            (config.tcp_host, config.tcp_port),
            family="AF_INET",
        )
    if config.socket_path:
        return create_sidecar_client(config.socket_path, family="AF_UNIX")
    raise RuntimeError("sidecar socket path is required for unix transport")


def describe_sidecar_service_config(
    config: SidecarServiceConfig | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    transport = _normalize_transport(config.transport)
    payload: dict[str, Any] = {
        "transport": transport,
        "worker_pool_mode": config.worker_pool_mode,
        "start_method": config.start_method,
        "worker_count": config.manager.workers.worker_count,
        "cpu_affinity_map": [
            list(cpu_ids) for cpu_ids in config.manager.workers.cpu_affinity_map or ()
        ],
        "control_cpu_affinity": list(config.control_cpu_affinity or ()),
        "reusable_cache_bytes": config.manager.cache.max_reusable_bytes,
        "reusable_ttl_s": config.manager.cache.reusable_entry_ttl_s,
    }
    if transport in {"auto", "unix"}:
        payload["socket_path"] = config.socket_path or _normalize_socket_path(None)
    if transport in {"auto", "tcp"}:
        payload["tcp_host"] = config.tcp_host
        payload["tcp_port"] = config.tcp_port
    return payload


__all__ = [
    "SidecarClient",
    "SidecarServiceConfig",
    "SidecarServiceProcess",
    "connect_sidecar_client_from_env",
    "create_sidecar_client",
    "describe_sidecar_service_config",
    "sidecar_service_config_from_env",
]
