from __future__ import annotations

import base64
import math
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from mm_sidecar.contracts import (
    ArtifactDescriptor,
    ImageScheduleItem,
    ImageTensorPayload,
    ProcessorSignature,
    StorageKind,
)

from .config import WorkerPoolConfig
from .protocol import FallbackDescriptor, PreparedArtifact, SidecarHandle


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass(frozen=True, slots=True)
class WorkerTask:
    cache_key: str
    epoch: int
    assigned_worker_id: int
    descriptor: FallbackDescriptor


@dataclass(frozen=True, slots=True)
class WorkerResult:
    cache_key: str
    epoch: int
    worker_id: int
    event_type: str
    at_ms: float
    schedule_item: ImageScheduleItem | None = None
    descriptor: ArtifactDescriptor | None = None
    payload: ImageTensorPayload | None = None
    timings_ms: dict[str, float] | None = None
    error_message: str | None = None


class ProcessorWorkerPool(Protocol):
    worker_count: int

    def submit(self, task: WorkerTask) -> None:
        ...

    def poll(self, max_items: int | None = None) -> list[WorkerResult]:
        ...

    def poll_ready(self, max_items: int | None = None) -> list[WorkerResult]:
        ...

    def close(self) -> None:
        ...


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv_floats(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not raw:
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return default


def _parse_signature(descriptor: FallbackDescriptor) -> dict[str, str]:
    return ProcessorSignature.parse(descriptor.processor_signature_value)


@dataclass(frozen=True, slots=True)
class _ImageProcessingPlan:
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    do_resize: bool
    do_rescale: bool
    do_normalize: bool
    do_convert_rgb: bool
    rescale_factor: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    orig_size_hw: tuple[int, int]
    resized_size_hw: tuple[int, int]
    image_grid_thw: tuple[int, int, int]

    @property
    def placeholder_token_count(self) -> int:
        grid_t, grid_h, grid_w = self.image_grid_thw
        return max(
            1,
            (int(grid_t) * int(grid_h) * int(grid_w)) // (self.merge_size**2),
        )


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _load_encoded_bytes(descriptor: FallbackDescriptor) -> bytes:
    source_ref = descriptor.captured_image.source_ref
    transport = source_ref.transport.value
    if transport == "local_path":
        local_path = source_ref.local_path
        if not local_path:
            raise ValueError("local_path descriptor requires local_path")
        return Path(local_path).read_bytes()
    if transport == "http":
        image_url = source_ref.image_url
        if not image_url:
            raise ValueError("http descriptor requires image_url")
        headers = {key: value for key, value in descriptor.http_headers}
        request = urllib.request.Request(image_url, headers=headers)
        timeout_s = descriptor.http_timeout_ms / 1000.0
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    if transport == "base64":
        image_url = source_ref.image_url
        if not image_url:
            raise ValueError("base64 descriptor requires image_url")
        if "," not in image_url:
            raise ValueError("base64 data url must contain a comma separator")
        payload = image_url.split(",", 1)[1]
        return base64.b64decode(payload, validate=False)
    raise ValueError(f"unsupported transport: {transport}")


def _load_image(encoded_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(encoded_bytes)) as image:
        image.load()
        return image.copy()


def _validate_limits(descriptor: FallbackDescriptor, encoded_bytes: bytes, image: Image.Image) -> None:
    limits = descriptor.ingress_limits
    limits.validate_encoded_bytes(len(encoded_bytes))
    limits.validate_pixel_count(image.width * image.height)
    limits.validate_decoded_bytes(image.width * image.height * len(image.getbands()))


def _bind_worker_cpu(cpu_affinity: tuple[int, ...] | None) -> None:
    if not cpu_affinity or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, set(cpu_affinity))
    except OSError:
        return


def _build_image_processing_plan(
    image: Image.Image,
    descriptor: FallbackDescriptor,
) -> _ImageProcessingPlan:
    signature = _parse_signature(descriptor)
    patch_size = int(signature.get("patch", "14"))
    merge_size = int(signature.get("merge", "2"))
    temporal_patch_size = int(signature.get("temporal", "1"))
    min_pixels = int(signature.get("min_pixels", str(56 * 56)))
    max_pixels = int(signature.get("max_pixels", str(28 * 28 * 1280)))
    do_resize = _parse_bool(signature.get("do_resize"), True)
    do_rescale = _parse_bool(signature.get("do_rescale"), True)
    do_normalize = _parse_bool(signature.get("do_normalize"), True)
    do_convert_rgb = _parse_bool(signature.get("do_convert_rgb"), True)
    rescale_factor = float(signature.get("rescale_factor", repr(1 / 255)))
    image_mean = _parse_csv_floats(
        signature.get("image_mean"),
        (0.48145466, 0.4578275, 0.40821073),
    )
    image_std = _parse_csv_floats(
        signature.get("image_std"),
        (0.26862954, 0.26130258, 0.27577711),
    )

    working = image.convert("RGB") if do_convert_rgb else image.copy()
    orig_h = int(working.height)
    orig_w = int(working.width)
    resized_h, resized_w = orig_h, orig_w
    if do_resize:
        resized_h, resized_w = _smart_resize(
            height=orig_h,
            width=orig_w,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    grid_t = max(1, 1 // temporal_patch_size)
    grid_h = max(1, resized_h // patch_size)
    grid_w = max(1, resized_w // patch_size)
    return _ImageProcessingPlan(
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        do_resize=do_resize,
        do_rescale=do_rescale,
        do_normalize=do_normalize,
        do_convert_rgb=do_convert_rgb,
        rescale_factor=rescale_factor,
        image_mean=image_mean,
        image_std=image_std,
        orig_size_hw=(orig_h, orig_w),
        resized_size_hw=(resized_h, resized_w),
        image_grid_thw=(int(grid_t), int(grid_h), int(grid_w)),
    )


def _build_schedule_item(
    descriptor: FallbackDescriptor,
    plan: _ImageProcessingPlan,
) -> ImageScheduleItem:
    return ImageScheduleItem(
        item_index=descriptor.request_media_index,
        item_identity=descriptor.item_identity,
        processor_signature=ProcessorSignature(
            value=descriptor.processor_signature_value
        ),
        orig_size_hw=plan.orig_size_hw,
        preprocessed_size_hw=plan.resized_size_hw,
        image_grid_thw=plan.image_grid_thw,
        placeholder_token_count=plan.placeholder_token_count,
    )


def _build_image_tensor_payload(
    image: Image.Image,
    descriptor: FallbackDescriptor,
    plan: _ImageProcessingPlan,
) -> tuple[ArtifactDescriptor, ImageTensorPayload]:
    working = image.convert("RGB") if plan.do_convert_rgb else image.copy()
    resized_h, resized_w = plan.resized_size_hw
    if plan.do_resize:
        working = working.resize(
            (resized_w, resized_h),
            resample=Image.Resampling.BICUBIC,
        )

    array = np.asarray(working)
    if plan.do_rescale:
        array = array.astype(np.float32) * plan.rescale_factor
    else:
        array = array.astype(np.float32)
    if plan.do_normalize:
        mean = np.asarray(plan.image_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(plan.image_std, dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / std

    array = np.transpose(array, (2, 0, 1))
    processed_images = np.expand_dims(array, axis=0)
    if processed_images.shape[0] % plan.temporal_patch_size != 0:
        repeats = np.repeat(
            processed_images[-1][np.newaxis],
            plan.temporal_patch_size - (processed_images.shape[0] % plan.temporal_patch_size),
            axis=0,
        )
        processed_images = np.concatenate([processed_images, repeats], axis=0)

    channel = processed_images.shape[1]
    grid_t, grid_h, grid_w = plan.image_grid_thw
    patches = processed_images.reshape(
        grid_t,
        plan.temporal_patch_size,
        channel,
        grid_h // plan.merge_size,
        plan.merge_size,
        plan.patch_size,
        grid_w // plan.merge_size,
        plan.merge_size,
        plan.patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    pixel_values = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * plan.temporal_patch_size * plan.patch_size * plan.patch_size,
    )

    descriptor_artifact = ArtifactDescriptor(
        artifact_id=f"{descriptor.cache_key}:epoch:{descriptor.request_media_index}",
        item_identity=descriptor.item_identity,
        processor_signature=ProcessorSignature(value=descriptor.processor_signature_value),
        image_grid_thw=plan.image_grid_thw,
        payload_shape=(int(pixel_values.shape[0]), int(pixel_values.shape[1])),
        payload_dtype=str(pixel_values.dtype),
        storage_kind=StorageKind.CPU_MEMORY,
        payload_nbytes=int(pixel_values.nbytes),
    )
    payload = ImageTensorPayload(
        pixel_values=pixel_values,
        image_grid_thw=plan.image_grid_thw,
        payload_shape=(int(pixel_values.shape[0]), int(pixel_values.shape[1])),
        payload_dtype=str(pixel_values.dtype),
        storage_kind=StorageKind.CPU_MEMORY,
        resized_size_hw=plan.resized_size_hw,
        orig_size_hw=plan.orig_size_hw,
        pixel_mean=float(np.mean(pixel_values)),
        pixel_std=float(np.std(pixel_values)),
    )
    return descriptor_artifact, payload


def _load_and_probe_task(
    task: WorkerTask,
) -> tuple[Image.Image, ImageScheduleItem, _ImageProcessingPlan, float, float, dict[str, float]]:
    started_at = time.perf_counter()
    after_source_started = time.perf_counter()
    encoded_bytes = _load_encoded_bytes(task.descriptor)
    after_source = time.perf_counter()

    image = _load_image(encoded_bytes)
    after_decode = time.perf_counter()

    _validate_limits(task.descriptor, encoded_bytes, image)
    plan = _build_image_processing_plan(image, task.descriptor)
    schedule_item = _build_schedule_item(task.descriptor, plan)
    after_probe = time.perf_counter()

    timings_ms = {
        "source": (after_source - after_source_started) * 1000.0,
        "decode": (after_decode - after_source) * 1000.0,
        "probe": (after_probe - after_decode) * 1000.0,
        "probe_ready_since_worker_start": (after_probe - started_at) * 1000.0,
    }
    return image, schedule_item, plan, started_at, after_probe, timings_ms


def _complete_payload_task(
    task: WorkerTask,
    image: Image.Image,
    plan: _ImageProcessingPlan,
    *,
    started_at: float,
    after_probe: float,
    timings_ms: dict[str, float],
) -> tuple[ArtifactDescriptor, ImageTensorPayload, dict[str, float]]:
    descriptor, payload = _build_image_tensor_payload(image, task.descriptor, plan)
    after_preprocess = time.perf_counter()
    completed_timings_ms = {
        **timings_ms,
        "preprocess": (after_preprocess - after_probe) * 1000.0,
        "total": (after_preprocess - started_at) * 1000.0,
    }
    return descriptor, payload, completed_timings_ms


def _run_task(
    task: WorkerTask,
) -> tuple[ImageScheduleItem, ArtifactDescriptor, ImageTensorPayload, dict[str, float]]:
    (
        image,
        schedule_item,
        plan,
        started_at,
        after_probe,
        timings_ms,
    ) = _load_and_probe_task(task)
    descriptor, payload, completed_timings_ms = _complete_payload_task(
        task,
        image,
        plan,
        started_at=started_at,
        after_probe=after_probe,
        timings_ms=timings_ms,
    )
    return schedule_item, descriptor, payload, completed_timings_ms


def run_descriptor_locally(
    descriptor: FallbackDescriptor,
    *,
    epoch: int = 0,
    worker_id: int = -1,
) -> PreparedArtifact:
    task = WorkerTask(
        cache_key=descriptor.cache_key,
        epoch=epoch,
        assigned_worker_id=worker_id,
        descriptor=descriptor,
    )
    _schedule_item, artifact_descriptor, payload, timings_ms = _run_task(task)
    return PreparedArtifact(
        handle=SidecarHandle(
            request_id=descriptor.request_id,
            request_media_index=descriptor.request_media_index,
            cache_key=descriptor.cache_key,
            epoch=epoch,
        ),
        descriptor=artifact_descriptor,
        payload=payload,
        timings_ms=timings_ms,
    )


def _worker_main(
    task_queue: "mp.queues.Queue[WorkerTask | None]",
    result_queue: "mp.queues.Queue[WorkerResult]",
    ready_result_queue: "mp.queues.Queue[WorkerResult] | None",
    worker_id: int,
    cpu_affinity: tuple[int, ...] | None,
) -> None:
    _bind_worker_cpu(cpu_affinity)
    while True:
        task = task_queue.get()
        if task is None:
            return
        worker_started_put_at_ms = _now_ms()
        result_queue.put(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=worker_id,
                event_type="started",
                at_ms=worker_started_put_at_ms,
                timings_ms={
                    "worker_started_put_at_ms": worker_started_put_at_ms,
                },
            )
        )
        try:
            (
                image,
                schedule_item,
                plan,
                started_at,
                after_probe,
                timings_ms,
            ) = _load_and_probe_task(task)
        except Exception as exc:
            result_queue.put(
                WorkerResult(
                    cache_key=task.cache_key,
                    epoch=task.epoch,
                    worker_id=worker_id,
                    event_type="failed",
                    at_ms=_now_ms(),
                    error_message=(
                        f"{exc.__class__.__name__}: {exc}\n"
                        f"{traceback.format_exc(limit=5)}"
                    ),
                )
            )
            continue
        worker_probed_put_at_ms = _now_ms()
        result_queue.put(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=worker_id,
                event_type="probed",
                at_ms=worker_probed_put_at_ms,
                schedule_item=schedule_item,
                timings_ms={
                    "source": timings_ms["source"],
                    "decode": timings_ms["decode"],
                    "probe": timings_ms["probe"],
                    "probe_ready_since_worker_start": timings_ms["probe_ready_since_worker_start"],
                    "worker_started_put_at_ms": worker_started_put_at_ms,
                    "worker_probed_put_at_ms": worker_probed_put_at_ms,
                    "worker_start_to_probed_put_ms": (
                        worker_probed_put_at_ms - worker_started_put_at_ms
                    ),
                },
            )
        )
        try:
            descriptor, payload, completed_timings_ms = _complete_payload_task(
                task,
                image,
                plan,
                started_at=started_at,
                after_probe=after_probe,
                timings_ms=timings_ms,
            )
        except Exception as exc:
            result_queue.put(
                WorkerResult(
                    cache_key=task.cache_key,
                    epoch=task.epoch,
                    worker_id=worker_id,
                    event_type="failed",
                    at_ms=_now_ms(),
                    schedule_item=schedule_item,
                    timings_ms=timings_ms,
                    error_message=(
                        f"{exc.__class__.__name__}: {exc}\n"
                        f"{traceback.format_exc(limit=5)}"
                    ),
                )
            )
            continue
        target_queue = ready_result_queue or result_queue
        worker_ready_put_start_at_ms = _now_ms()
        ready_timings_ms = {
            **completed_timings_ms,
            "worker_ready_put_start_at_ms": worker_ready_put_start_at_ms,
            "worker_ready_payload_nbytes": float(payload.nbytes),
        }
        target_queue.put(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=worker_id,
                event_type="ready",
                at_ms=worker_ready_put_start_at_ms,
                schedule_item=schedule_item,
                descriptor=descriptor,
                payload=payload,
                timings_ms=ready_timings_ms,
            )
        )
        worker_ready_put_done_at_ms = _now_ms()
        result_queue.put(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=worker_id,
                event_type="ready_put_done",
                at_ms=worker_ready_put_done_at_ms,
                timings_ms={
                    "worker_ready_put_start_at_ms": worker_ready_put_start_at_ms,
                    "worker_ready_put_done_at_ms": worker_ready_put_done_at_ms,
                    "worker_ready_put_call_ms": (
                        worker_ready_put_done_at_ms - worker_ready_put_start_at_ms
                    ),
                    "worker_ready_payload_nbytes": float(payload.nbytes),
                },
            )
        )


class InlineProcessorWorkerPool:
    def __init__(self, worker_count: int = 1) -> None:
        self.worker_count = worker_count
        self._results: list[WorkerResult] = []
        self.submission_count = 0

    def submit(self, task: WorkerTask) -> None:
        self.submission_count += 1
        self._results.append(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=task.assigned_worker_id,
                event_type="started",
                at_ms=_now_ms(),
            )
        )
        try:
            (
                image,
                schedule_item,
                plan,
                started_at,
                after_probe,
                timings_ms,
            ) = _load_and_probe_task(task)
        except Exception as exc:
            self._results.append(
                WorkerResult(
                    cache_key=task.cache_key,
                    epoch=task.epoch,
                    worker_id=task.assigned_worker_id,
                    event_type="failed",
                    at_ms=_now_ms(),
                    error_message=f"{exc.__class__.__name__}: {exc}",
                )
            )
            return
        self._results.append(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=task.assigned_worker_id,
                event_type="probed",
                at_ms=_now_ms(),
                schedule_item=schedule_item,
                timings_ms={
                    "source": timings_ms["source"],
                    "decode": timings_ms["decode"],
                    "probe": timings_ms["probe"],
                },
            )
        )
        try:
            descriptor, payload, completed_timings_ms = _complete_payload_task(
                task,
                image,
                plan,
                started_at=started_at,
                after_probe=after_probe,
                timings_ms=timings_ms,
            )
        except Exception as exc:
            self._results.append(
                WorkerResult(
                    cache_key=task.cache_key,
                    epoch=task.epoch,
                    worker_id=task.assigned_worker_id,
                    event_type="failed",
                    at_ms=_now_ms(),
                    schedule_item=schedule_item,
                    timings_ms=timings_ms,
                    error_message=f"{exc.__class__.__name__}: {exc}",
                )
            )
            return
        self._results.append(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=task.assigned_worker_id,
                event_type="ready",
                at_ms=_now_ms(),
                schedule_item=schedule_item,
                descriptor=descriptor,
                payload=payload,
                timings_ms={
                    **completed_timings_ms,
                    "worker_ready_put_call_ms": 0.0,
                    "worker_ready_payload_nbytes": float(payload.nbytes),
                },
            )
        )

    def poll(self, max_items: int | None = None) -> list[WorkerResult]:
        if max_items is None or max_items >= len(self._results):
            results = list(self._results)
            self._results.clear()
            return results
        results = self._results[:max_items]
        del self._results[:max_items]
        return results

    def poll_ready(self, max_items: int | None = None) -> list[WorkerResult]:
        return self.poll(max_items=max_items)

    def close(self) -> None:
        self._results.clear()


class MultiProcessProcessorWorkerPool:
    def __init__(self, config: WorkerPoolConfig | None = None) -> None:
        self._config = config or WorkerPoolConfig()
        self.worker_count = self._config.worker_count
        self._ctx = mp.get_context(self._config.start_method)
        self._result_queue: "mp.queues.Queue[WorkerResult]" = self._ctx.Queue()
        self._ready_result_queue: "mp.queues.Queue[WorkerResult]" = self._ctx.Queue()
        self._task_queues: list["mp.queues.Queue[WorkerTask | None]"] = []
        self._processes: list[mp.Process] = []
        self._closed = False

        for worker_id in range(self.worker_count):
            task_queue: "mp.queues.Queue[WorkerTask | None]" = self._ctx.Queue()
            cpu_affinity = None
            if self._config.cpu_affinity_map and worker_id < len(self._config.cpu_affinity_map):
                cpu_affinity = self._config.cpu_affinity_map[worker_id]
            process = self._ctx.Process(
                target=_worker_main,
                args=(
                    task_queue,
                    self._result_queue,
                    self._ready_result_queue,
                    worker_id,
                    cpu_affinity,
                ),
                daemon=True,
            )
            process.start()
            self._task_queues.append(task_queue)
            self._processes.append(process)

    def submit(self, task: WorkerTask) -> None:
        if self._closed:
            raise RuntimeError("worker pool is closed")
        self._task_queues[task.assigned_worker_id].put(task)

    def poll(self, max_items: int | None = None) -> list[WorkerResult]:
        limit = max_items if max_items is not None else self.worker_count * 16
        results: list[WorkerResult] = []
        while len(results) < limit:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def poll_ready(self, max_items: int | None = None) -> list[WorkerResult]:
        limit = max_items if max_items is not None else self.worker_count * 4
        results: list[WorkerResult] = []
        while len(results) < limit:
            try:
                results.append(self._ready_result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task_queue in self._task_queues:
            task_queue.put(None)
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._ready_result_queue.get_nowait()
            except queue.Empty:
                break
