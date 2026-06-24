#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from PIL import Image

from mm_sidecar.contracts import (
    CapturedImageRef,
    IngressLimits,
    MediaTransport,
    NormalizedImage,
    ProcessorConfig,
    ProcessorSignature,
)
from mm_sidecar.contracts.media_source import MediaSourceRef
from mm_sidecar.sidecar import (
    MemoryCacheConfig,
    MultiProcessProcessorWorkerPool,
    SidecarManager,
    SidecarManagerConfig,
    SidecarState,
    WorkerPoolConfig,
)
from mm_sidecar.sidecar.protocol import FallbackDescriptor, SidecarHandle


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def maximum(values: list[float]) -> float:
    return max(values) if values else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg_ms": average(values),
        "max_ms": maximum(values),
    }


def build_signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="/autodl-fs/data/qwen3.5-0.8b",
            revision="unknown",
            processor_name="Qwen3_5Config",
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=784,
            max_pixels=1_003_520,
            do_resize=True,
        )
    )


def build_limits() -> IngressLimits:
    return IngressLimits(
        max_image_count=40,
        max_encoded_bytes=64 * 1024 * 1024,
        max_decoded_bytes=512 * 1024 * 1024,
        max_pixels_per_image=1280 * 28 * 28,
    )


def available_cpu_ids() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
        if cpu_ids:
            return cpu_ids
    return tuple(range(os.cpu_count() or 1))


def build_affinity_map(worker_count: int) -> tuple[tuple[int, ...], ...]:
    cpu_ids = available_cpu_ids()
    return tuple((cpu_ids[index % len(cpu_ids)],) for index in range(worker_count))


def wait_for_any_ready(
    manager: SidecarManager,
    handles: list[SidecarHandle],
    timeout_ms: float,
    poll_interval_ms: float = 1.0,
) -> float:
    deadline = time.perf_counter() + timeout_ms / 1000.0
    while True:
        snapshots = manager.batch_get_status(handles)
        if any(snapshot.state is SidecarState.READY for snapshot in snapshots):
            return time.perf_counter()
        if time.perf_counter() >= deadline:
            raise TimeoutError("no handle became READY before timeout")
        time.sleep(poll_interval_ms / 1000.0)


def build_captured_image(normalized: NormalizedImage) -> CapturedImageRef:
    return CapturedImageRef(
        source_ref=normalized.source_ref,
        mime_type=normalized.mime_type,
        byte_size=normalized.byte_size,
        local_materialized_path=normalized.local_materialized_path,
    )


def build_local_descriptors(
    *,
    image_paths: list[Path],
    round_id: int,
    orig_size_hw: tuple[int, int],
    signature: ProcessorSignature,
    limits: IngressLimits,
) -> list[FallbackDescriptor]:
    request_id = f"req-local-{round_id}"
    descriptors: list[FallbackDescriptor] = []
    for item_index, image_path in enumerate(image_paths):
        stat_result = image_path.stat()
        identity = (
            f"local_path:{image_path.resolve()}|{stat_result.st_mtime_ns}|"
            f"{stat_result.st_size}|round:{round_id}"
        )
        normalized = NormalizedImage(
            source_ref=MediaSourceRef(
                transport=MediaTransport.LOCAL_PATH,
                source_key=identity,
                media_uuid=f"uuid-local-{round_id}-{item_index}",
                request_scope_key=None,
                local_path=str(image_path.resolve()),
            ),
            orig_size_hw=orig_size_hw,
            mime_type="image/jpeg",
            byte_size=int(stat_result.st_size),
            decoded_size_hw=orig_size_hw,
            local_materialized_path=str(image_path.resolve()),
        )
        descriptors.append(
            FallbackDescriptor(
                request_id=request_id,
                request_media_index=item_index,
                captured_image=build_captured_image(normalized),
                ingress_limits=limits,
                processor_signature_value=signature.value,
                item_identity=identity,
                orig_size_hw=orig_size_hw,
            )
        )
    return descriptors


def build_http_descriptors(
    *,
    base_url: str,
    image_count: int,
    round_id: int,
    orig_size_hw: tuple[int, int],
    signature: ProcessorSignature,
    limits: IngressLimits,
) -> list[FallbackDescriptor]:
    request_id = f"req-http-{round_id}"
    descriptors: list[FallbackDescriptor] = []
    for item_index in range(image_count):
        request_url = f"{base_url}?round={round_id}&item={item_index}"
        identity = f"http:{request_url}"
        normalized = NormalizedImage(
            source_ref=MediaSourceRef(
                transport=MediaTransport.HTTP,
                source_key=identity,
                media_uuid=f"uuid-http-{round_id}-{item_index}",
                request_scope_key=None,
                image_url=request_url,
            ),
            orig_size_hw=orig_size_hw,
            mime_type="image/jpeg",
            byte_size=None,
            decoded_size_hw=orig_size_hw,
        )
        descriptors.append(
            FallbackDescriptor(
                request_id=request_id,
                request_media_index=item_index,
                captured_image=build_captured_image(normalized),
                ingress_limits=limits,
                processor_signature_value=signature.value,
                item_identity=identity,
                orig_size_hw=orig_size_hw,
                http_timeout_ms=5_000,
            )
        )
    return descriptors


def build_base64_descriptors(
    *,
    image_bytes: bytes,
    image_count: int,
    round_id: int,
    orig_size_hw: tuple[int, int],
    signature: ProcessorSignature,
    limits: IngressLimits,
) -> list[FallbackDescriptor]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    request_id = f"req-base64-{round_id}"
    request_scope_key = request_id
    image_url = f"data:image/jpeg;base64,{encoded}"
    descriptors: list[FallbackDescriptor] = []
    for item_index in range(image_count):
        identity = f"base64:{request_scope_key}:image:{item_index}"
        normalized = NormalizedImage(
            source_ref=MediaSourceRef(
                transport=MediaTransport.BASE64,
                source_key=identity,
                media_uuid=f"uuid-base64-{round_id}-{item_index}",
                request_scope_key=request_scope_key,
                image_url=image_url,
                mime_type="image/jpeg",
            ),
            orig_size_hw=orig_size_hw,
            mime_type="image/jpeg",
            byte_size=len(image_bytes),
            decoded_size_hw=orig_size_hw,
        )
        descriptors.append(
            FallbackDescriptor(
                request_id=request_id,
                request_media_index=item_index,
                captured_image=build_captured_image(normalized),
                ingress_limits=limits,
                processor_signature_value=signature.value,
                item_identity=identity,
                orig_size_hw=orig_size_hw,
            )
        )
    return descriptors


def fetch_all_ready(manager: SidecarManager, handles: list[SidecarHandle]) -> int:
    total_payload_bytes = 0
    for handle in handles:
        artifact = manager.fetch_ready(handle)
        if artifact is None:
            raise RuntimeError("fetch_ready returned None")
        total_payload_bytes += artifact.descriptor.payload_nbytes or 0
    return total_payload_bytes


def run_batch_benchmark(
    *,
    transport: str,
    manager: SidecarManager,
    descriptor_builder,
    warmup: int,
    rounds: int,
) -> dict[str, object]:
    metrics: dict[str, list[float]] = {
        "prepare_ms": [],
        "first_ready_after_prepare_ms": [],
        "all_ready_after_prepare_ms": [],
        "fetch_all_ms": [],
        "cold_e2e_ms": [],
        "hot_prepare_ms": [],
        "hot_fetch_all_ms": [],
        "hot_e2e_ms": [],
        "payload_nbytes": [],
    }

    for round_id in range(warmup + rounds):
        descriptors = descriptor_builder(round_id)

        t0 = time.perf_counter()
        handles = list(manager.prepare(descriptors))
        t1 = time.perf_counter()
        first_ready_at = wait_for_any_ready(manager, handles, timeout_ms=5_000.0)
        snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 5_000.0)
        t2 = time.perf_counter()
        if any(snapshot.state is not SidecarState.READY for snapshot in snapshots):
            raise RuntimeError(f"{transport} batch did not fully become READY")
        total_payload_bytes = fetch_all_ready(manager, handles)
        t3 = time.perf_counter()

        hot_t0 = time.perf_counter()
        hot_handles = list(manager.prepare(descriptors))
        hot_t1 = time.perf_counter()
        hot_payload_bytes = fetch_all_ready(manager, hot_handles)
        hot_t2 = time.perf_counter()
        if hot_payload_bytes != total_payload_bytes:
            raise RuntimeError(f"{transport} cache-hit payload bytes mismatch")

        if round_id >= warmup:
            metrics["prepare_ms"].append((t1 - t0) * 1000.0)
            metrics["first_ready_after_prepare_ms"].append((first_ready_at - t1) * 1000.0)
            metrics["all_ready_after_prepare_ms"].append((t2 - t1) * 1000.0)
            metrics["fetch_all_ms"].append((t3 - t2) * 1000.0)
            metrics["cold_e2e_ms"].append((t3 - t0) * 1000.0)
            metrics["hot_prepare_ms"].append((hot_t1 - hot_t0) * 1000.0)
            metrics["hot_fetch_all_ms"].append((hot_t2 - hot_t1) * 1000.0)
            metrics["hot_e2e_ms"].append((hot_t2 - hot_t0) * 1000.0)
            metrics["payload_nbytes"].append(float(total_payload_bytes))

    return {
        "prepare_ms": summarize(metrics["prepare_ms"]),
        "first_ready_after_prepare_ms": summarize(metrics["first_ready_after_prepare_ms"]),
        "all_ready_after_prepare_ms": summarize(metrics["all_ready_after_prepare_ms"]),
        "fetch_all_ms": summarize(metrics["fetch_all_ms"]),
        "cold_e2e_ms": summarize(metrics["cold_e2e_ms"]),
        "hot_prepare_ms": summarize(metrics["hot_prepare_ms"]),
        "hot_fetch_all_ms": summarize(metrics["hot_fetch_all_ms"]),
        "hot_e2e_ms": summarize(metrics["hot_e2e_ms"]),
        "payload_nbytes": int(statistics.mean(metrics["payload_nbytes"])) if metrics["payload_nbytes"] else 0,
    }


def render_table(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, dict)
    lines = []
    lines.append("Stage C Sidecar Runtime Batch")
    lines.append(
        "| transport | image_count | cold_prepare | first_ready_after_prepare | all_ready_after_prepare | fetch_all | cold_e2e | hot_prepare | hot_fetch_all | hot_e2e | payload_bytes |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    image_count = payload["config"]["image_count"]
    for transport, result in results.items():
        assert isinstance(result, dict)
        lines.append(
            f"| {transport} | {image_count} | "
            f"{result['prepare_ms']['avg_ms']:.3f} / {result['prepare_ms']['max_ms']:.3f} | "
            f"{result['first_ready_after_prepare_ms']['avg_ms']:.3f} / {result['first_ready_after_prepare_ms']['max_ms']:.3f} | "
            f"{result['all_ready_after_prepare_ms']['avg_ms']:.3f} / {result['all_ready_after_prepare_ms']['max_ms']:.3f} | "
            f"{result['fetch_all_ms']['avg_ms']:.3f} / {result['fetch_all_ms']['max_ms']:.3f} | "
            f"{result['cold_e2e_ms']['avg_ms']:.3f} / {result['cold_e2e_ms']['max_ms']:.3f} | "
            f"{result['hot_prepare_ms']['avg_ms']:.3f} / {result['hot_prepare_ms']['max_ms']:.3f} | "
            f"{result['hot_fetch_all_ms']['avg_ms']:.3f} / {result['hot_fetch_all_ms']['max_ms']:.3f} | "
            f"{result['hot_e2e_ms']['avg_ms']:.3f} / {result['hot_e2e_ms']['max_ms']:.3f} | "
            f"{result['payload_nbytes']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--http-url", required=True)
    parser.add_argument("--image-count", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image_path.resolve()
    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as image:
        orig_size_hw = (image.height, image.width)

    signature = build_signature()
    limits = build_limits()
    affinity_map = build_affinity_map(args.workers)
    worker_config = WorkerPoolConfig(
        worker_count=args.workers,
        cpu_affinity_map=affinity_map,
        start_method="fork",
    )

    with tempfile.TemporaryDirectory(prefix="sidecar_runtime_batch_") as tmpdir:
        local_dir = Path(tmpdir) / "local_images"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_paths: list[Path] = []
        for item_index in range(args.image_count):
            dst = local_dir / f"stagec_batch_{item_index:02d}.jpg"
            shutil.copy2(image_path, dst)
            local_paths.append(dst)

        manager = SidecarManager(
            config=SidecarManagerConfig(
                cache=MemoryCacheConfig(max_reusable_bytes=512 * 1024 * 1024),
                workers=worker_config,
            ),
            worker_pool=MultiProcessProcessorWorkerPool(worker_config),
        )
        try:
            payload = {
                "config": {
                    "image_path": str(image_path),
                    "http_url": args.http_url,
                    "image_count": args.image_count,
                    "warmup": args.warmup,
                    "rounds": args.rounds,
                    "workers": args.workers,
                    "cpu_affinity_preview": [list(item) for item in affinity_map[:8]],
                    "orig_size_hw": list(orig_size_hw),
                },
                "results": {
                    "local_path": run_batch_benchmark(
                        transport="local_path",
                        manager=manager,
                        descriptor_builder=lambda round_id: build_local_descriptors(
                            image_paths=local_paths,
                            round_id=round_id,
                            orig_size_hw=orig_size_hw,
                            signature=signature,
                            limits=limits,
                        ),
                        warmup=args.warmup,
                        rounds=args.rounds,
                    ),
                    "http": run_batch_benchmark(
                        transport="http",
                        manager=manager,
                        descriptor_builder=lambda round_id: build_http_descriptors(
                            base_url=args.http_url,
                            image_count=args.image_count,
                            round_id=round_id,
                            orig_size_hw=orig_size_hw,
                            signature=signature,
                            limits=limits,
                        ),
                        warmup=args.warmup,
                        rounds=args.rounds,
                    ),
                    "base64": run_batch_benchmark(
                        transport="base64",
                        manager=manager,
                        descriptor_builder=lambda round_id: build_base64_descriptors(
                            image_bytes=image_bytes,
                            image_count=args.image_count,
                            round_id=round_id,
                            orig_size_hw=orig_size_hw,
                            signature=signature,
                            limits=limits,
                        ),
                        warmup=args.warmup,
                        rounds=args.rounds,
                    ),
                },
            }
        finally:
            manager.close()

    if args.output is not None:
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(render_table(payload))
    print("")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
