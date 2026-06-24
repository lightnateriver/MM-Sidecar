#!/usr/bin/env python3
"""Benchmark sidecar-like manifest extraction for Qwen3.5/Qwen3VL images.

This script does not start a vLLM server. It benchmarks a sidecar-like CPU
metadata path across three transports:

- local_path
- http
- base64

It compares four manifest granularities:

- hw_only
- grid_thw_only
- grid_thw_plus_token_count
- full_manifest

Correctness is validated against a slower reference path that fully opens the
image with Pillow before computing the same Qwen3VL-style grid/token outputs.

The `grid_thw` / token-count path is aligned to the vLLM Qwen3VL source:

- `factor = patch_size * spatial_merge_size`
- `grid_h = resized_height // patch_size`
- `grid_w = resized_width // patch_size`
- `num_tokens = int(prod(grid_thw)) // (merge_size ** 2)`

When available, the script imports the exact `smart_resize` implementation from
`qwen_vl_utils` or `transformers`. Otherwise it falls back to a mirrored
implementation that follows the public Qwen2VL/Qwen3VL behavior closely.
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shutil
import socketserver
import statistics
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import build_opener, urlopen

from PIL import Image

MODES = (
    "hw_only",
    "grid_thw_only",
    "grid_thw_plus_token_count",
    "full_manifest",
)

TRANSPORTS = ("local_path", "http", "base64")

DEFAULT_MIN_PIXELS = 56 * 56
DEFAULT_MAX_PIXELS = 12845056

WORKER_CONTEXT: dict[str, Any] = {}
WORKER_AFFINITY: dict[str, Any] = {
    "pid": None,
    "worker_slot": None,
    "assigned_cpu": None,
    "status": "not_initialized",
    "before": None,
    "after": None,
    "reason": "",
}
WORKER_HTTP_OPENER = None


@dataclass
class ProbeConfig:
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    channels: int = 3
    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    do_resize: bool = True

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @property
    def pixel_payload_width(self) -> int:
        return (
            self.channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "avg_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_probe_config(model_dir: Path | None) -> tuple[ProbeConfig, dict[str, Any]]:
    config = ProbeConfig()
    details: dict[str, Any] = {
        "model_dir": str(model_dir) if model_dir else None,
        "config_json_found": False,
        "preprocessor_config_found": False,
        "sources": [],
    }

    if model_dir is None:
        details["sources"].append("builtin_defaults")
        return config, details

    config_json = model_dir / "config.json"
    preprocessor_json = model_dir / "preprocessor_config.json"

    cfg_data = read_json_if_exists(config_json)
    if cfg_data:
        details["config_json_found"] = True
        vision_cfg = cfg_data.get("vision_config", {})
        if isinstance(vision_cfg, dict):
            if isinstance(vision_cfg.get("patch_size"), int):
                config.patch_size = int(vision_cfg["patch_size"])
            if isinstance(vision_cfg.get("spatial_merge_size"), int):
                config.merge_size = int(vision_cfg["spatial_merge_size"])
            if isinstance(vision_cfg.get("temporal_patch_size"), int):
                config.temporal_patch_size = int(vision_cfg["temporal_patch_size"])
            if isinstance(vision_cfg.get("in_channels"), int):
                config.channels = int(vision_cfg["in_channels"])
        details["sources"].append("config.json")

    proc_data = read_json_if_exists(preprocessor_json)
    if proc_data:
        details["preprocessor_config_found"] = True
        size = proc_data.get("size")
        if isinstance(size, dict):
            if isinstance(size.get("shortest_edge"), int):
                config.min_pixels = int(size["shortest_edge"])
            if isinstance(size.get("longest_edge"), int):
                config.max_pixels = int(size["longest_edge"])
        if isinstance(proc_data.get("min_pixels"), int):
            config.min_pixels = int(proc_data["min_pixels"])
        if isinstance(proc_data.get("max_pixels"), int):
            config.max_pixels = int(proc_data["max_pixels"])
        details["sources"].append("preprocessor_config.json")

    if not details["sources"]:
        details["sources"].append("builtin_defaults")
    return config, details


def smart_resize_fallback(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    **_: Any,
) -> tuple[int, int]:
    if height < 1 or width < 1:
        raise ValueError(f"invalid image size: {width}x{height}")
    max_ratio = max(height, width) / min(height, width)
    if max_ratio > 200:
        raise ValueError(f"extreme aspect ratio not supported: {width}x{height}")

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)

    area = resized_height * resized_width
    if area > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(
            factor, math.floor(height / beta / factor) * factor
        )
        resized_width = max(
            factor, math.floor(width / beta / factor) * factor
        )
    elif area < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = max(
            factor, math.ceil(height * beta / factor) * factor
        )
        resized_width = max(
            factor, math.ceil(width * beta / factor) * factor
        )

    return int(resized_height), int(resized_width)


def resolve_smart_resize() -> tuple[Any, dict[str, Any]]:
    try:
        from qwen_vl_utils import smart_resize  # type: ignore

        return smart_resize, {
            "backend": "qwen_vl_utils.smart_resize",
            "exact_reference": True,
        }
    except Exception:
        pass

    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (  # type: ignore
            smart_resize,
        )

        return smart_resize, {
            "backend": "transformers.qwen2_vl.smart_resize",
            "exact_reference": True,
        }
    except Exception:
        pass

    return smart_resize_fallback, {
        "backend": "mirrored_fallback",
        "exact_reference": False,
    }


def create_test_images(root: Path, count: int, width: int, height: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(count):
        path = root / f"img_{idx:02d}.jpg"
        if not path.exists():
            rng = random.Random(20260617 + idx)
            image = Image.new("RGB", (width, height))
            pixels = [
                (
                    rng.randrange(256),
                    rng.randrange(256),
                    rng.randrange(256),
                )
                for _ in range(width * height)
            ]
            image.putdata(pixels)
            image.save(path, format="JPEG", quality=90)
        paths.append(path)
    return paths


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def copyfile(self, source: Any, outputfile: Any) -> None:
        try:
            shutil.copyfileobj(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            return


class QuietThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


class LocalHTTPServer:
    def __init__(self, serve_dir: Path, port: int = 0) -> None:
        handler = functools.partial(
            QuietHTTPRequestHandler,
            directory=str(serve_dir),
        )
        self._server = QuietThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="manifest-contract-bench-http",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def wait_http_ready(base_url: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(base_url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"HTTP server not ready: {base_url}")


def read_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return None


def configure_worker_affinity(
    worker_slot: int,
    assigned_cpu: int | None,
    affinity_enabled: bool,
) -> dict[str, Any]:
    global WORKER_AFFINITY
    global WORKER_HTTP_OPENER

    before = read_affinity()
    after = before
    status = "disabled"
    reason = "affinity_disabled_or_unavailable"

    if affinity_enabled and assigned_cpu is not None:
        try:
            os.sched_setaffinity(0, {assigned_cpu})
            after = read_affinity()
            if after == [assigned_cpu]:
                status = "bound"
                reason = "ok"
            elif after and assigned_cpu in after:
                status = "partially_bound"
                reason = "assigned_cpu_present_but_affinity_has_extra_cpus"
            else:
                status = "not_bound"
                reason = "assigned_cpu_not_in_effective_affinity"
        except Exception as exc:
            after = read_affinity()
            status = "not_bound"
            reason = f"sched_setaffinity_error:{type(exc).__name__}:{exc}"

    WORKER_AFFINITY = {
        "pid": os.getpid(),
        "worker_slot": worker_slot,
        "assigned_cpu": assigned_cpu,
        "status": status,
        "before": before,
        "after": after,
        "reason": reason,
    }
    WORKER_HTTP_OPENER = build_opener()
    return dict(WORKER_AFFINITY)


def build_affinity_plan(n_workers: int, bind_cpu: bool) -> dict[str, Any]:
    if not bind_cpu:
        return {
            "enabled": False,
            "available_cpus": [],
            "assignments": [],
            "independent": False,
            "reason": "disabled_by_cli",
        }

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {
            "enabled": False,
            "available_cpus": [],
            "assignments": [],
            "independent": False,
            "reason": "os_sched_affinity_unavailable",
        }

    available_cpus = read_affinity()
    if not available_cpus:
        return {
            "enabled": False,
            "available_cpus": [],
            "assignments": [],
            "independent": False,
            "reason": "could_not_read_available_cpu_list",
        }

    independent = len(available_cpus) >= n_workers
    assignments = [
        available_cpus[idx]
        if idx < len(available_cpus)
        else available_cpus[idx % len(available_cpus)]
        for idx in range(n_workers)
    ]
    return {
        "enabled": True,
        "available_cpus": available_cpus,
        "assignments": assignments,
        "independent": independent,
        "reason": "ok" if independent else "fewer_available_cpus_than_workers",
    }


def init_worker_context(
    probe_config: dict[str, Any],
    resize_backend: str,
    exact_reference: bool,
) -> None:
    smart_resize, _ = resolve_smart_resize()
    WORKER_CONTEXT.clear()
    WORKER_CONTEXT.update(
        {
            "probe_config": ProbeConfig(**probe_config),
            "resize_backend": resize_backend,
            "smart_resize": smart_resize,
            "exact_reference": exact_reference,
        }
    )


def parse_jpeg_size_partial(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4:
        return None
    if data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG stream")

    i = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    standalone = {0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7}

    while i + 1 < len(data):
        while i < len(data) and data[i] != 0xFF:
            i += 1
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            return None

        marker = data[i]
        i += 1

        if marker == 0xD9:
            return None
        if marker in standalone:
            continue
        if i + 1 >= len(data):
            return None

        seg_len = (data[i] << 8) | data[i + 1]
        if seg_len < 2:
            raise ValueError(f"invalid JPEG segment length: {seg_len}")

        if marker in sof_markers:
            if i + 6 >= len(data):
                return None
            height = (data[i + 3] << 8) | data[i + 4]
            width = (data[i + 5] << 8) | data[i + 6]
            return width, height

        i += seg_len

    return None


def probe_jpeg_size_from_file(path: str, chunk_size: int = 8192) -> tuple[int, int]:
    data = bytearray()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data.extend(chunk)
            size = parse_jpeg_size_partial(data)
            if size is not None:
                return size

    with Image.open(path) as image:
        return image.size


def probe_jpeg_size_from_bytes(data: bytes) -> tuple[int, int]:
    size = parse_jpeg_size_partial(data)
    if size is not None:
        return size
    with Image.open(BytesIO(data)) as image:
        return image.size


def probe_jpeg_size_from_http(url: str, chunk_size: int = 8192) -> tuple[int, int]:
    global WORKER_HTTP_OPENER
    opener = WORKER_HTTP_OPENER or build_opener()
    data = bytearray()
    with opener.open(url, timeout=30) as resp:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            data.extend(chunk)
            size = parse_jpeg_size_partial(data)
            if size is not None:
                return size
    return probe_jpeg_size_from_bytes(bytes(data))


def probe_jpeg_size_from_base64(payload: str, chunk_chars: int = 4096) -> tuple[int, int]:
    prefix, encoded = split_base64_payload(payload)
    data = bytearray()
    carry = ""

    for idx in range(0, len(encoded), chunk_chars):
        carry += encoded[idx : idx + chunk_chars]
        usable = (len(carry) // 4) * 4
        if usable == 0:
            continue
        piece = carry[:usable]
        carry = carry[usable:]
        decoded = base64.b64decode(piece, validate=False)
        data.extend(decoded)
        size = parse_jpeg_size_partial(data)
        if size is not None:
            return size

    if carry:
        data.extend(base64.b64decode(carry, validate=False))
    if prefix:
        _ = prefix
    return probe_jpeg_size_from_bytes(bytes(data))


def split_base64_payload(payload: str) -> tuple[str, str]:
    if payload.startswith("data:") and "," in payload:
        prefix, encoded = payload.split(",", 1)
        return prefix, encoded
    return "", payload


def full_decode_base64(payload: str) -> bytes:
    _, encoded = split_base64_payload(payload)
    return base64.b64decode(encoded, validate=False)


def build_item_identity(source: dict[str, Any]) -> str:
    transport = source["transport"]
    if transport == "local_path":
        path = Path(source["path"])
        st = path.stat()
        return f"local:{path.name}:{st.st_size}:{st.st_mtime_ns}"
    if transport == "http":
        return f"http:{source['url']}"
    if transport == "base64":
        payload = source["base64"]
        head = payload[:32]
        tail = payload[-32:] if len(payload) > 32 else payload
        return f"base64:{len(payload)}:{head}:{tail}"
    raise ValueError(f"unknown transport: {transport}")


def compute_qwen3vl_metadata(
    width: int,
    height: int,
    probe_config: ProbeConfig,
    smart_resize: Any,
) -> dict[str, Any]:
    if probe_config.do_resize:
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=probe_config.factor,
            min_pixels=probe_config.min_pixels,
            max_pixels=probe_config.max_pixels,
        )
    else:
        resized_height, resized_width = height, width

    grid_t = 1
    grid_h = int(resized_height) // probe_config.patch_size
    grid_w = int(resized_width) // probe_config.patch_size
    token_count = int(grid_t * grid_h * grid_w) // (probe_config.merge_size**2)

    return {
        "width": int(width),
        "height": int(height),
        "preprocessed_width": int(resized_width),
        "preprocessed_height": int(resized_height),
        "image_grid_thw": [grid_t, grid_h, grid_w],
        "placeholder_token_count": int(token_count),
        "payload_shape": [
            int(grid_t * grid_h * grid_w),
            int(probe_config.pixel_payload_width),
        ],
        "payload_dtype": "float32",
    }


def processor_signature(
    probe_config: ProbeConfig,
    resize_backend: str,
    source_name: str,
) -> str:
    return (
        f"{source_name}:patch={probe_config.patch_size}:merge={probe_config.merge_size}:"
        f"temporal={probe_config.temporal_patch_size}:channels={probe_config.channels}:"
        f"min_pixels={probe_config.min_pixels}:max_pixels={probe_config.max_pixels}:"
        f"resize_backend={resize_backend}:do_resize={int(probe_config.do_resize)}"
    )


def materialize_manifest(
    mode: str,
    metadata: dict[str, Any],
    probe_config: ProbeConfig,
    resize_backend: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    if mode == "hw_only":
        return {
            "width": metadata["width"],
            "height": metadata["height"],
        }

    if mode == "grid_thw_only":
        return {
            "image_grid_thw": metadata["image_grid_thw"],
        }

    if mode == "grid_thw_plus_token_count":
        return {
            "image_grid_thw": metadata["image_grid_thw"],
            "placeholder_token_count": metadata["placeholder_token_count"],
        }

    if mode == "full_manifest":
        return {
            "image_grid_thw": metadata["image_grid_thw"],
            "placeholder_token_count": metadata["placeholder_token_count"],
            "processor_signature": processor_signature(
                probe_config, resize_backend, "qwen3vl_like"
            ),
            "payload_shape": metadata["payload_shape"],
            "payload_dtype": metadata["payload_dtype"],
            "item_identity": build_item_identity(source),
        }

    raise ValueError(f"unknown mode: {mode}")


def extract_manifest_fast(source: dict[str, Any]) -> dict[str, Any]:
    probe_config: ProbeConfig = WORKER_CONTEXT["probe_config"]
    smart_resize = WORKER_CONTEXT["smart_resize"]
    resize_backend = WORKER_CONTEXT["resize_backend"]
    transport = source["transport"]

    if transport == "local_path":
        width, height = probe_jpeg_size_from_file(source["path"])
    elif transport == "http":
        width, height = probe_jpeg_size_from_http(source["url"])
    elif transport == "base64":
        width, height = probe_jpeg_size_from_base64(source["base64"])
    else:
        raise ValueError(f"unknown transport: {transport}")

    metadata = compute_qwen3vl_metadata(width, height, probe_config, smart_resize)
    return materialize_manifest(source["mode"], metadata, probe_config, resize_backend, source)


def extract_manifest_reference(
    source: dict[str, Any],
    probe_config: ProbeConfig,
    smart_resize: Any,
    resize_backend: str,
) -> dict[str, Any]:
    transport = source["transport"]

    if transport == "local_path":
        with Image.open(source["path"]) as image:
            width, height = image.size
    elif transport == "http":
        with urlopen(source["url"], timeout=30) as resp:
            data = resp.read()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    elif transport == "base64":
        data = full_decode_base64(source["base64"])
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    else:
        raise ValueError(f"unknown transport: {transport}")

    metadata = compute_qwen3vl_metadata(width, height, probe_config, smart_resize)
    return materialize_manifest(source["mode"], metadata, probe_config, resize_backend, source)


def worker_run_benchmark_round(payload: dict[str, Any]) -> dict[str, Any]:
    start_at = float(payload["start_at"])
    items = payload["items"]

    delay = start_at - time.perf_counter()
    if delay > 0:
        time.sleep(delay)

    item_completion_ms: list[float] = []
    item_elapsed_ms: list[float] = []
    for source in items:
        item_start = time.perf_counter()
        extract_manifest_fast(source)
        item_elapsed_ms.append((time.perf_counter() - item_start) * 1000.0)
        item_completion_ms.append((time.perf_counter() - start_at) * 1000.0)

    return {
        "worker_pid": os.getpid(),
        "worker_slot": WORKER_AFFINITY.get("worker_slot"),
        "assigned_cpu": WORKER_AFFINITY.get("assigned_cpu"),
        "affinity_status": WORKER_AFFINITY.get("status"),
        "items_processed": len(items),
        "item_completion_ms": item_completion_ms,
        "item_elapsed_ms": item_elapsed_ms,
    }


def persistent_worker_main(
    worker_slot: int,
    assigned_cpu: int | None,
    affinity_enabled: bool,
    probe_config: dict[str, Any],
    resize_backend: str,
    exact_reference: bool,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        init_worker_context(probe_config, resize_backend, exact_reference)
        affinity = configure_worker_affinity(worker_slot, assigned_cpu, affinity_enabled)
        result_queue.put(
            {
                "type": "ready",
                "worker_slot": worker_slot,
                "affinity": affinity,
            }
        )
        while True:
            task = task_queue.get()
            task_type = task.get("type")
            if task_type == "stop":
                return
            if task_type == "benchmark":
                result = worker_run_benchmark_round(task)
            else:
                raise RuntimeError(f"unknown task type: {task_type}")
            result_queue.put(
                {
                    "type": "result",
                    "round_id": task.get("round_id"),
                    "worker_slot": worker_slot,
                    "result": result,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "worker_slot": worker_slot,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


class PersistentWorkerPool:
    def __init__(
        self,
        assignments: list[int],
        affinity_enabled: bool,
        n_workers: int,
        probe_config: ProbeConfig,
        resize_backend: str,
        exact_reference: bool,
        start_method: str = "auto",
        startup_timeout_s: float = 10.0,
    ) -> None:
        ctx = mp.get_context(resolve_start_method(start_method))
        self._result_queue: Any = ctx.Queue()
        self._task_queues: list[Any] = []
        self._processes: list[mp.Process] = []
        self._round_id = 0
        self.worker_affinity: list[dict[str, Any]] = []

        for idx in range(n_workers):
            worker_slot = idx + 1
            assigned_cpu = assignments[idx] if idx < len(assignments) else None
            task_queue: Any = ctx.Queue()
            process = ctx.Process(
                target=persistent_worker_main,
                args=(
                    worker_slot,
                    assigned_cpu,
                    affinity_enabled,
                    asdict(probe_config),
                    resize_backend,
                    exact_reference,
                    task_queue,
                    self._result_queue,
                ),
                name=f"manifest-bench-worker-{worker_slot}",
            )
            process.start()
            self._task_queues.append(task_queue)
            self._processes.append(process)

        self.worker_affinity = self._collect_ready(n_workers, startup_timeout_s)

    def __enter__(self) -> "PersistentWorkerPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        for task_queue in self._task_queues:
            try:
                task_queue.put({"type": "stop"})
            except Exception:
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def _collect_ready(self, expected: int, timeout_s: float) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        deadline = time.time() + timeout_s
        while len(reports) < expected and time.time() < deadline:
            try:
                message = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get("type") == "ready":
                reports.append(message["affinity"])
            elif message.get("type") == "error":
                raise RuntimeError(
                    f"worker {message.get('worker_slot')} failed during startup: "
                    f"{message.get('error')}\n{message.get('traceback')}"
                )
        if len(reports) != expected:
            raise RuntimeError(f"expected {expected} workers, got {len(reports)} ready reports")
        return sorted(reports, key=lambda item: item.get("worker_slot", -1))

    def run_round(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(payloads) != len(self._task_queues):
            raise ValueError(
                f"expected {len(self._task_queues)} task payloads, got {len(payloads)}"
            )

        self._round_id += 1
        round_id = self._round_id
        for task_queue, payload in zip(self._task_queues, payloads):
            task_queue.put({"type": "benchmark", "round_id": round_id, **payload})

        results: list[dict[str, Any]] = []
        deadline = time.time() + 120.0
        while len(results) < len(payloads) and time.time() < deadline:
            try:
                message = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get("type") == "result" and message.get("round_id") == round_id:
                results.append(message["result"])
            elif message.get("type") == "error":
                raise RuntimeError(
                    f"worker {message.get('worker_slot')} failed: "
                    f"{message.get('error')}\n{message.get('traceback')}"
                )
        if len(results) != len(payloads):
            raise RuntimeError(
                f"expected {len(payloads)} worker results for round {round_id}, got {len(results)}"
            )
        return sorted(results, key=lambda item: item.get("worker_slot", -1))


def build_affinity_summary(
    affinity_plan: dict[str, Any],
    worker_affinity: list[dict[str, Any]],
    mp_workers: int,
) -> dict[str, Any]:
    bound_workers = sum(1 for item in worker_affinity if item["status"] == "bound")
    assigned_cpus = [item["assigned_cpu"] for item in worker_affinity]
    return {
        "plan": affinity_plan,
        "workers": worker_affinity,
        "bound_workers": bound_workers,
        "all_workers_bound": len(worker_affinity) == mp_workers and bound_workers == mp_workers,
        "independent_cpu_binding": (
            bool(affinity_plan["independent"])
            and len(worker_affinity) == mp_workers
            and len(set(assigned_cpus)) == mp_workers
            and bound_workers == mp_workers
        ),
    }


def chunk_round_robin(items: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for idx, item in enumerate(items):
        chunks[idx % workers].append(item)
    return chunks


def run_mode_round(
    pool: PersistentWorkerPool,
    items: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    start_at = time.perf_counter() + 0.25
    chunks = chunk_round_robin(items, workers)
    payloads = [{"start_at": start_at, "items": chunk} for chunk in chunks]
    worker_results = pool.run_round(payloads)

    completion_ms = [
        value
        for worker in worker_results
        for value in worker["item_completion_ms"]
    ]
    if not completion_ms:
        raise RuntimeError("no item completion timestamps collected")

    return {
        "first_item_ready_ms": min(completion_ms),
        "all_items_ready_ms": max(completion_ms),
        "worker_results": worker_results,
    }


def benchmark_mode(
    pool: PersistentWorkerPool,
    items: list[dict[str, Any]],
    workers: int,
    warmup: int,
    rounds: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        run_mode_round(pool, items, workers)

    measured = [run_mode_round(pool, items, workers) for _ in range(rounds)]
    first_values = [item["first_item_ready_ms"] for item in measured]
    all_values = [item["all_items_ready_ms"] for item in measured]

    return {
        "summary": {
            "first_item_ready_ms": summarize_metric(first_values),
            "all_items_ready_ms": summarize_metric(all_values),
        },
        "rounds": measured,
    }


def source_for_transport(
    transport: str,
    path: Path,
    base_url: str,
    b64_payload: str,
) -> dict[str, Any]:
    if transport == "local_path":
        return {
            "transport": transport,
            "path": str(path),
        }
    if transport == "http":
        return {
            "transport": transport,
            "url": f"{base_url}/{path.name}",
        }
    if transport == "base64":
        return {
            "transport": transport,
            "base64": b64_payload,
        }
    raise ValueError(f"unknown transport: {transport}")


def build_scenario_sources(
    image_paths: list[Path],
    base_url: str,
) -> dict[str, list[dict[str, Any]]]:
    encoded_payloads: list[str] = []
    for path in image_paths:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        encoded_payloads.append(f"data:image/jpeg;base64,{encoded}")

    scenarios: dict[str, list[dict[str, Any]]] = {}
    for transport in TRANSPORTS:
        scenarios[transport] = [
            source_for_transport(transport, path, base_url, payload)
            for path, payload in zip(image_paths, encoded_payloads)
        ]
    return scenarios


def validate_scenario(
    transport: str,
    image_count: int,
    base_sources: list[dict[str, Any]],
    probe_config: ProbeConfig,
    smart_resize: Any,
    resize_backend: str,
) -> dict[str, Any]:
    mode_results: dict[str, Any] = {}
    sample_fast = None
    sample_reference = None

    for mode in MODES:
        fast_items = []
        reference_items = []
        mismatches = []
        for base_source in base_sources[:image_count]:
            source = dict(base_source)
            source["mode"] = mode

            fast = extract_manifest_reference(
                {"transport": source["transport"], **{k: v for k, v in source.items() if k != "mode"}, "mode": mode},
                probe_config,
                smart_resize,
                resize_backend,
            )
            reference = extract_manifest_reference(
                source,
                probe_config,
                smart_resize,
                resize_backend,
            )
            fast_items.append(fast)
            reference_items.append(reference)
            if fast != reference:
                mismatches.append({"fast": fast, "reference": reference})

        if sample_fast is None and fast_items:
            sample_fast = fast_items[0]
            sample_reference = reference_items[0]

        mode_results[mode] = {
            "all_match": not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
        }

    return {
        "transport": transport,
        "image_count": image_count,
        "all_modes_match": all(result["all_match"] for result in mode_results.values()),
        "modes": mode_results,
        "sample_fast": sample_fast,
        "sample_reference": sample_reference,
    }


def validate_scenario_with_fast_probe(
    transport: str,
    image_count: int,
    base_sources: list[dict[str, Any]],
    probe_config: ProbeConfig,
    smart_resize: Any,
    resize_backend: str,
) -> dict[str, Any]:
    mode_results: dict[str, Any] = {}
    sample_fast = None
    sample_reference = None

    for mode in MODES:
        mismatches = []
        for base_source in base_sources[:image_count]:
            source = dict(base_source)
            source["mode"] = mode
            fast = extract_manifest_fast_local(source, probe_config, smart_resize, resize_backend)
            reference = extract_manifest_reference(
                source,
                probe_config,
                smart_resize,
                resize_backend,
            )
            if sample_fast is None:
                sample_fast = fast
                sample_reference = reference
            if fast != reference:
                mismatches.append({"fast": fast, "reference": reference})

        mode_results[mode] = {
            "all_match": not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
        }

    return {
        "transport": transport,
        "image_count": image_count,
        "all_modes_match": all(result["all_match"] for result in mode_results.values()),
        "modes": mode_results,
        "sample_fast": sample_fast,
        "sample_reference": sample_reference,
    }


def extract_manifest_fast_local(
    source: dict[str, Any],
    probe_config: ProbeConfig,
    smart_resize: Any,
    resize_backend: str,
) -> dict[str, Any]:
    transport = source["transport"]
    if transport == "local_path":
        width, height = probe_jpeg_size_from_file(source["path"])
    elif transport == "http":
        width, height = probe_jpeg_size_from_http(source["url"])
    elif transport == "base64":
        width, height = probe_jpeg_size_from_base64(source["base64"])
    else:
        raise ValueError(f"unknown transport: {transport}")

    metadata = compute_qwen3vl_metadata(width, height, probe_config, smart_resize)
    return materialize_manifest(source["mode"], metadata, probe_config, resize_backend, source)


def render_tables(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Correctness")
    lines.append("| transport | image_count | all_modes_match | reference_backend | exact_reference |")
    lines.append("|---|---:|---|---|---|")
    for scenario in results["scenarios"]:
        correctness = scenario["correctness"]
        lines.append(
            f"| {scenario['transport']} | {scenario['image_count']} | "
            f"{'PASS' if correctness['all_modes_match'] else 'FAIL'} | "
            f"{results['resize_backend']['backend']} | "
            f"{'yes' if results['resize_backend']['exact_reference'] else 'no'} |"
        )

    lines.append("")
    lines.append("All Items Ready")
    lines.append("| transport | image_count | hw_only | grid_thw_only | grid+token | full_manifest |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for scenario in results["scenarios"]:
        parts = []
        for mode in MODES:
            summary = scenario["modes"][mode]["summary"]["all_items_ready_ms"]
            parts.append(f"{summary['avg_ms']:.3f} / {summary['max_ms']:.3f}")
        lines.append(
            f"| {scenario['transport']} | {scenario['image_count']} | "
            + " | ".join(parts)
            + " |"
        )

    lines.append("")
    lines.append("First Item Ready")
    lines.append("| transport | image_count | hw_only | grid_thw_only | grid+token | full_manifest |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for scenario in results["scenarios"]:
        parts = []
        for mode in MODES:
            summary = scenario["modes"][mode]["summary"]["first_item_ready_ms"]
            parts.append(f"{summary['avg_ms']:.3f} / {summary['max_ms']:.3f}")
        lines.append(
            f"| {scenario['transport']} | {scenario['image_count']} | "
            + " | ".join(parts)
            + " |"
        )

    return "\n".join(lines)


def resolve_start_method(start_method: str) -> str:
    if start_method == "auto":
        available = mp.get_all_start_methods()
        if "fork" in available:
            return "fork"
        return "spawn"
    return start_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--counts", type=str, default="1,13,20")
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--bind-cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/manifest_contract_bench.json"))
    parser.add_argument("--http-port", type=int, default=0)
    parser.add_argument("--worker-start-method", type=str, default="auto")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()

    counts = [int(item) for item in args.counts.split(",") if item.strip()]
    if not counts:
        raise ValueError("counts must not be empty")
    max_count = max(counts)

    probe_config, probe_config_details = load_probe_config(args.model_dir)
    smart_resize, resize_backend = resolve_smart_resize()

    with tempfile.TemporaryDirectory(prefix="manifest_contract_bench_") as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_paths = create_test_images(image_dir, max_count, args.width, args.height)

        server = LocalHTTPServer(image_dir, args.http_port)
        server.start()
        try:
            base_url = f"http://127.0.0.1:{server.port}"
            wait_http_ready(f"{base_url}/")
            scenario_sources = build_scenario_sources(image_paths, base_url)

            affinity_plan = build_affinity_plan(args.workers, args.bind_cpu)
            with PersistentWorkerPool(
                assignments=affinity_plan["assignments"],
                affinity_enabled=bool(affinity_plan["enabled"]),
                n_workers=args.workers,
                probe_config=probe_config,
                resize_backend=resize_backend["backend"],
                exact_reference=bool(resize_backend["exact_reference"]),
                start_method=args.worker_start_method,
                startup_timeout_s=args.startup_timeout,
            ) as pool:
                affinity_summary = build_affinity_summary(
                    affinity_plan,
                    pool.worker_affinity,
                    args.workers,
                )

                scenarios: list[dict[str, Any]] = []
                for transport in TRANSPORTS:
                    for image_count in counts:
                        base_sources = scenario_sources[transport][:image_count]
                        correctness = validate_scenario_with_fast_probe(
                            transport=transport,
                            image_count=image_count,
                            base_sources=base_sources,
                            probe_config=probe_config,
                            smart_resize=smart_resize,
                            resize_backend=resize_backend["backend"],
                        )

                        mode_results: dict[str, Any] = {}
                        for mode in MODES:
                            items = [
                                {**source, "mode": mode}
                                for source in base_sources
                            ]
                            mode_results[mode] = benchmark_mode(
                                pool=pool,
                                items=items,
                                workers=args.workers,
                                warmup=args.warmup,
                                rounds=args.rounds,
                            )

                        scenarios.append(
                            {
                                "transport": transport,
                                "image_count": image_count,
                                "correctness": correctness,
                                "modes": mode_results,
                            }
                        )
        finally:
            server.stop()

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "width": args.width,
            "height": args.height,
            "counts": counts,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "workers": args.workers,
            "bind_cpu": args.bind_cpu,
            "worker_start_method": resolve_start_method(args.worker_start_method),
            "startup_timeout": args.startup_timeout,
        },
        "probe_config": asdict(probe_config),
        "probe_config_details": probe_config_details,
        "resize_backend": resize_backend,
        "affinity": affinity_summary,
        "scenarios": scenarios,
    }

    write_json(args.output, payload)
    print(render_tables(payload))
    print("")
    print(f"JSON written to: {args.output}")


if __name__ == "__main__":
    main()
