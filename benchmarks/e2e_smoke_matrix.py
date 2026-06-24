#!/usr/bin/env python3
"""Run a small multimodal transport smoke matrix against a vLLM service.

The script is intentionally self-contained so it can be copied to the remote
server with the mm-sidecar project and reused for both stock vLLM and patched
vLLM services.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import hashlib
import http.server
import json
import socket
import socketserver
import statistics
import threading
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Scenario:
    transport: str
    image_count: int


@dataclass
class StaticServer:
    server: socketserver.TCPServer
    thread: threading.Thread
    base_url: str

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


@dataclass(frozen=True)
class FixtureImage:
    path: Path
    expected_color: str


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="OpenAI API server root, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", default="auto", help="Model id. Use 'auto' to read /v1/models.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-counts", default="1,13,20")
    parser.add_argument("--transports", default="local_path,http,base64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--media-port", type=int, default=19080)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--fetch-debug", action="store_true")
    parser.add_argument("--request-prefix", default="mm-sidecar-smoke")
    parser.add_argument("--image-seed", type=int, default=20260623)
    return parser.parse_args()


def ensure_free_port(host: str, preferred_port: int) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind((host, preferred_port))
            return preferred_port
        except OSError:
            pass
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def start_static_server(root: Path, preferred_port: int) -> StaticServer:
    port = ensure_free_port("127.0.0.1", preferred_port)
    handler = functools.partial(QuietHandler, directory=str(root))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return StaticServer(
        server=server,
        thread=thread,
        base_url=f"http://127.0.0.1:{port}",
    )


def get_model_id(host: str, requested_model: str, timeout: int) -> str:
    if requested_model != "auto":
        return requested_model
    response = requests.get(f"{host}/v1/models", timeout=timeout)
    response.raise_for_status()
    data = response.json().get("data") or []
    if not data:
        raise RuntimeError("/v1/models returned no model ids")
    return str(data[0]["id"])


def generate_fixture_images(root: Path, max_images: int, width: int, height: int) -> list[Path]:
    image_dir = root / "fixtures" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    palette = [
        (216, 32, 40),
        (24, 112, 220),
        (32, 156, 92),
        (236, 178, 32),
        (172, 76, 188),
    ]
    for index in range(max_images):
        color = palette[index % len(palette)]
        image = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, width - 10, 70), fill=(255, 255, 255))
        draw.text((24, 28), f"MM SIDECAR {index:02d}", fill=(0, 0, 0))
        path = image_dir / f"img_{index:02d}.jpg"
        image.save(path, format="JPEG", quality=92, optimize=True)
        paths.append(path)
    return paths


def generate_random_fixture_images(
    root: Path,
    total_images: int,
    width: int,
    height: int,
    *,
    seed: int,
) -> list[FixtureImage]:
    image_dir = root / "fixtures" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fixtures: list[FixtureImage] = []
    palette = [
        ("red", (216, 32, 40)),
        ("blue", (24, 112, 220)),
        ("green", (32, 156, 92)),
        ("yellow", (236, 178, 32)),
        ("purple", (172, 76, 188)),
        ("orange", (232, 124, 32)),
    ]
    for index in range(total_images):
        expected_color, background = palette[index % len(palette)]
        image = Image.new(
            "RGB",
            (width, height),
            color=background,
        )
        draw = ImageDraw.Draw(image)
        stripe_count = 6
        stripe_height = max(1, height // stripe_count)
        for stripe_index in range(stripe_count):
            top = stripe_index * stripe_height
            bottom = height if stripe_index == stripe_count - 1 else min(
                height,
                (stripe_index + 1) * stripe_height,
            )
            overlay_fill = tuple(
                min(
                    255,
                    max(
                        0,
                        channel + rng.randint(-36, 36),
                    ),
                )
                for channel in background
            )
            draw.rectangle(
                (0, top, width, bottom),
                fill=overlay_fill,
            )
        for _ in range(24):
            x0 = rng.randint(0, max(0, width - 1))
            y0 = rng.randint(0, max(0, height - 1))
            x1 = rng.randint(x0, width)
            y1 = rng.randint(y0, height)
            draw.rectangle(
                (x0, y0, x1, y1),
                outline=(
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                ),
                width=max(1, min(width, height) // 96),
            )
        for _ in range(12):
            x0 = rng.randint(0, max(0, width - 1))
            y0 = rng.randint(0, max(0, height - 1))
            x1 = rng.randint(x0, width)
            y1 = rng.randint(y0, height)
            draw.ellipse(
                (x0, y0, x1, y1),
                outline=(
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                ),
                width=max(1, min(width, height) // 96),
            )
        draw.rectangle((10, 10, width - 10, 70), fill=(255, 255, 255))
        draw.text(
            (24, 28),
            f"MM RAND {index:04d} {expected_color.upper()}",
            fill=(0, 0, 0),
        )
        path = image_dir / f"img_{index:04d}.jpg"
        image.save(path, format="JPEG", quality=92, optimize=True)
        fixtures.append(FixtureImage(path=path, expected_color=expected_color))
    return fixtures


def allocate_case_images(
    images: list[FixtureImage],
    *,
    transports: list[str],
    image_counts: list[int],
    total_runs: int,
) -> dict[tuple[str, int, int], list[FixtureImage]]:
    allocation: dict[tuple[str, int, int], list[FixtureImage]] = {}
    cursor = 0
    for transport in transports:
        for image_count in image_counts:
            for run_id in range(total_runs):
                next_cursor = cursor + image_count
                if next_cursor > len(images):
                    raise RuntimeError(
                        "not enough images allocated for strict per-case unique runs"
                    )
                allocation[(transport, image_count, run_id)] = images[cursor:next_cursor]
                cursor = next_cursor
    return allocation


def image_url_for_transport(path: Path, root: Path, transport: str, http_base_url: str) -> str:
    if transport == "local_path":
        return path.resolve().as_uri()
    if transport == "http":
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        return f"{http_base_url.rstrip('/')}/{relative}"
    if transport == "base64":
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    raise ValueError(f"unsupported transport: {transport}")


def build_payload(
    *,
    model: str,
    image_paths: list[Path],
    root: Path,
    transport: str,
    http_base_url: str,
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Look at the first attached image only. "
                "What is its dominant color? Answer with one English color word."
            ),
        }
    ]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url_for_transport(path, root, transport, http_base_url)
                },
            }
        )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def stream_chat(
    host: str,
    payload: dict[str, Any],
    *,
    request_id: str,
    timeout: int,
    expected_color: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    final_chunk: dict[str, Any] | None = None
    headers: dict[str, str] = {}
    status_code = 0
    with requests.post(
        f"{host}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
        headers={"x-request-id": request_id},
    ) as response:
        status_code = response.status_code
        headers = dict(response.headers)
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            final_chunk = chunk
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(str(content))
    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    text = "".join(pieces)
    normalized = text.strip().lower()
    return {
        "status_code": status_code,
        "ttft_ms": (first_token_at - start) * 1000.0,
        "e2e_ms": (end - start) * 1000.0,
        "completion_text": text,
        "completion_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "response_headers": headers,
        "usage": (final_chunk or {}).get("usage") or {},
        "expected_color": expected_color,
        "semantic_pass": normalized == expected_color,
    }


def fetch_debug(host: str, timeout: int) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{host}/mm_sidecar/debug/last_capture", timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "max": 0.0}
    return {"avg": statistics.mean(values), "max": max(values)}


def render_markdown(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| transport | image_count | success | semantic | ttft avg/max ms | e2e avg/max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {transport} | {image_count} | {success}/{measured_runs} | "
            "{semantic}/{measured_runs} | {ttft_avg:.2f}/{ttft_max:.2f} | "
            "{e2e_avg:.2f}/{e2e_max:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    host = args.host.rstrip("/")
    image_counts = [int(item) for item in args.image_counts.split(",") if item.strip()]
    transports = [item.strip() for item in args.transports.split(",") if item.strip()]
    total_runs = args.warmup + args.runs
    total_required_images = sum(image_counts) * len(transports) * total_runs

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images = generate_random_fixture_images(
        args.work_dir,
        total_required_images,
        args.width,
        args.height,
        seed=args.image_seed,
    )
    case_images = allocate_case_images(
        images,
        transports=transports,
        image_counts=image_counts,
        total_runs=total_runs,
    )
    static_server = start_static_server(args.work_dir, args.media_port)
    try:
        model_id = get_model_id(host, args.model, args.timeout)
        records: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        for transport in transports:
            for image_count in image_counts:
                scenario = Scenario(transport=transport, image_count=image_count)
                measured: list[dict[str, Any]] = []
                for run_id in range(total_runs):
                    request_id = (
                        f"{args.request_prefix}-{transport}-{image_count}-run-{run_id}"
                    )
                    run_fixtures = case_images[(transport, image_count, run_id)]
                    run_images = [fixture.path for fixture in run_fixtures]
                    expected_color = run_fixtures[0].expected_color
                    payload = build_payload(
                        model=model_id,
                        image_paths=run_images,
                        root=args.work_dir,
                        transport=transport,
                        http_base_url=static_server.base_url,
                        max_tokens=args.max_tokens,
                    )
                    result = stream_chat(
                        host,
                        payload,
                        request_id=request_id,
                        timeout=args.timeout,
                        expected_color=expected_color,
                    )
                    debug_capture = fetch_debug(host, args.timeout) if args.fetch_debug else None
                    record = {
                        "transport": scenario.transport,
                        "image_count": scenario.image_count,
                        "run": run_id,
                        "is_warmup": run_id < args.warmup,
                        "request_id": request_id,
                        "expected_color": expected_color,
                        "case_image_paths": [str(path.resolve()) for path in run_images],
                        **result,
                        "debug_capture": debug_capture,
                    }
                    records.append(record)
                    if not record["is_warmup"]:
                        measured.append(record)
                    print(
                        f"transport={transport} images={image_count} "
                        f"run={run_id} warmup={record['is_warmup']} "
                        f"status={record['status_code']} "
                        f"semantic={record['semantic_pass']} "
                        f"ttft_ms={record['ttft_ms']:.2f} "
                        f"e2e_ms={record['e2e_ms']:.2f} "
                        f"text={record['completion_text'][:60]!r}",
                        flush=True,
                    )

                ttft = summarize([float(item["ttft_ms"]) for item in measured])
                e2e = summarize([float(item["e2e_ms"]) for item in measured])
                summary_rows.append(
                    {
                        "transport": transport,
                        "image_count": image_count,
                        "measured_runs": len(measured),
                        "success": sum(1 for item in measured if item["status_code"] == 200),
                        "semantic": sum(1 for item in measured if item["semantic_pass"]),
                        "ttft_avg": ttft["avg"],
                        "ttft_max": ttft["max"],
                        "e2e_avg": e2e["avg"],
                        "e2e_max": e2e["max"],
                    }
                )

        output = {
            "host": host,
            "model": model_id,
            "work_dir": str(args.work_dir.resolve()),
            "http_base_url": static_server.base_url,
            "image_size": {"width": args.width, "height": args.height},
            "warmup": args.warmup,
            "runs": args.runs,
            "total_runs_per_case": total_runs,
            "total_required_images": total_required_images,
            "image_seed": args.image_seed,
            "summary": summary_rows,
            "records": records,
        }
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        table_path = args.out.with_suffix(".md")
        table_path.write_text(render_markdown(summary_rows), encoding="utf-8")
        print(render_markdown(summary_rows))
        print(f"json={args.out}")
        print(f"table={table_path}")
    finally:
        static_server.shutdown()


if __name__ == "__main__":
    main()
