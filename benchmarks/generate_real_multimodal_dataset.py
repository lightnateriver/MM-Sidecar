#!/usr/bin/env python3
"""Generate a cache-safe real-image multimodal benchmark dataset.

This script is maintained inside the mm-sidecar project and builds a
deterministic real-image dataset for service benchmarking.

Design goals:
- exact text token target per request
- real photographic images instead of synthetic shapes
- unique final image bytes across rounds to reduce accidental MM cache hits
- local file payloads that are directly consumable by an OpenAI-compatible
  multimodal chat service
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from PIL import Image, ImageOps
from transformers import AutoTokenizer


LIST_URL = "https://picsum.photos/v2/list?page=1&limit=100"
USER_AGENT = "mm-sidecar-real-bench/1.0"


@dataclass(frozen=True)
class SourcePhoto:
    photo_id: str
    author: str
    width: int | None
    height: int | None
    download_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a cache-safe multimodal benchmark dataset with exact "
            "text token targets and real photographic images."
        )
    )
    parser.add_argument("--tokenizer", required=True, help="Model or tokenizer path")
    parser.add_argument("--request-model", required=True, help="Model field written into payloads")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8, help="Total request groups")
    parser.add_argument("--warmup-rounds", type=int, default=3, help="Warmup request groups")
    parser.add_argument("--images-per-round", type=int, default=13)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--target-text-tokens", type=int, default=10000)
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument(
        "--source-list-url",
        default=LIST_URL,
        help="JSON endpoint that returns a list of source photos",
    )
    parser.add_argument(
        "--source-urls-file",
        type=Path,
        help="Optional JSON or text file with one explicit image URL per line",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than 0")
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.rounds:
        raise ValueError("--warmup-rounds must be in [0, rounds)")
    if args.images_per_round <= 0:
        raise ValueError("--images-per-round must be greater than 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than 0")
    if args.target_text_tokens <= 0:
        raise ValueError("--target-text-tokens must be greater than 0")


def build_sentence(round_id: int, sentence_id: int) -> str:
    code = f"r{round_id:02d}_s{sentence_id:05d}"
    adjectives = [
        "granular",
        "deterministic",
        "cacheless",
        "multimodal",
        "latency-aware",
        "token-precise",
        "vision-heavy",
        "nonrepeating",
    ]
    nouns = [
        "benchmark",
        "request",
        "payload",
        "profile",
        "timeline",
        "dataset",
        "session",
        "sample",
    ]
    adj = adjectives[sentence_id % len(adjectives)]
    noun = nouns[(sentence_id * 3) % len(nouns)]
    numbers = [str(round_id), str(sentence_id), str(round_id * 1000 + sentence_id)]
    return (
        f"Segment {code} records a {adj} {noun} for service tracing. "
        f"It includes identifiers {'/'.join(numbers)} and unique markers "
        f"{code.upper()}::{sentence_id * 17 + round_id}. "
        f"This line is intentionally unique within and across rounds.\n"
    )


def fit_tail_exact(
    tokenizer: AutoTokenizer,
    prefix: str,
    round_id: int,
    sentence_id: int,
    target_tokens: int,
) -> str:
    current = len(tokenizer.encode(prefix, add_special_tokens=False))
    if current == target_tokens:
        return prefix

    def candidate_texts(seed: int) -> list[str]:
        return [
            f" tail{round_id:02d}_{sentence_id:05d}_{seed:05d}",
            f" item-{round_id:02d}-{sentence_id:05d}-{seed:05d}",
            f" note[{round_id:02d}:{sentence_id:05d}:{seed:05d}]",
            f" ref{seed:05d}",
            f" z{seed:05d}",
            f"\nextra_{round_id:02d}_{sentence_id:05d}_{seed:05d}",
            f" code={round_id * 100000 + sentence_id * 97 + seed}",
            f" flag{seed:05d}.",
        ]

    remaining = target_tokens - current
    stack: list[tuple[str, int, int]] = [(prefix, remaining, 1)]
    text = None

    while stack:
        base_text, base_remaining, seed = stack.pop()
        if base_remaining == 0:
            text = base_text
            break
        if base_remaining < 0 or seed > 4096:
            continue

        current_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
        options: list[tuple[int, str]] = []
        for fragment in candidate_texts(seed):
            merged = base_text + fragment
            merged_tokens = len(tokenizer.encode(merged, add_special_tokens=False))
            delta = merged_tokens - current_tokens
            if 0 < delta <= base_remaining:
                options.append((delta, merged))

        options.sort(key=lambda item: (-item[0], len(item[1])))
        stack.append((base_text, base_remaining, seed + 1))
        for delta, merged in reversed(options):
            stack.append((merged, base_remaining - delta, seed + 1))

    if text is None:
        raise RuntimeError(
            f"failed to fit exact token count for round={round_id}, remaining={remaining}"
        )

    final_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if final_tokens != target_tokens:
        raise RuntimeError(
            f"final token count mismatch: got={final_tokens}, target={target_tokens}"
        )
    return text


def build_text(
    tokenizer: AutoTokenizer,
    round_id: int,
    target_tokens: int,
) -> tuple[str, int]:
    random.seed(1000 + round_id)
    text = ""
    sentence_id = 0
    while True:
        candidate = text + build_sentence(round_id, sentence_id)
        candidate_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
        if candidate_tokens > target_tokens:
            text = fit_tail_exact(
                tokenizer,
                text,
                round_id,
                sentence_id,
                target_tokens,
            )
            break
        text = candidate
        sentence_id += 1

    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count != target_tokens:
        raise RuntimeError(
            f"unexpected token count for round {round_id}: {token_count}"
        )
    return text, token_count


def validate_text_uniqueness(text: str, round_id: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        raise RuntimeError(f"round_{round_id} generated duplicate text lines")
    return lines


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def urlopen_bytes(url: str) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme == "file":
        local_path = Path(url2pathname((parsed.netloc or "") + parsed.path))
        return local_path.read_bytes()
    if parsed.scheme == "" and Path(url).exists():
        return Path(url).read_bytes()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as resp:
        return resp.read()


def fetch_source_catalog(list_url: str, needed: int) -> list[SourcePhoto]:
    raw = urlopen_bytes(list_url)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("source list response is not a JSON array")

    sources: list[SourcePhoto] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            source = SourcePhoto(
                photo_id=str(item["id"]),
                author=str(item.get("author", "unknown")),
                width=int(item["width"]),
                height=int(item["height"]),
                download_url=str(item["download_url"]),
            )
        except Exception:
            continue

        if source.width < 900 or source.height < 900:
            continue
        sources.append(source)
        if len(sources) >= needed:
            break

    if len(sources) < needed:
        raise RuntimeError(
            f"not enough usable source photos in catalog: got={len(sources)}, need={needed}"
        )
    return sources


def fetch_explicit_sources(urls_file: Path, needed: int) -> list[SourcePhoto]:
    raw = urls_file.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            urls = [str(item).strip() for item in payload if str(item).strip()]
        else:
            raise ValueError("JSON payload is not a list")
    except Exception:
        urls = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#")]

    if len(urls) < needed:
        raise RuntimeError(
            f"source URL file has only {len(urls)} urls, need at least {needed}"
        )

    return [
        SourcePhoto(
            photo_id=f"explicit-{index:02d}",
            author="explicit",
            width=None,
            height=None,
            download_url=url,
        )
        for index, url in enumerate(urls[:needed])
    ]


def download_source_images(
    source_dir: Path,
    sources: list[SourcePhoto],
) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, source in enumerate(sources):
        raw = urlopen_bytes(source.download_url)
        suffix = ".jpg"
        path = source_dir / f"{index:02d}_id{source.photo_id}{suffix}"
        path.write_bytes(raw)
        paths.append(path)
        print(f"downloaded source photo id={source.photo_id} author={source.author} -> {path}")
    return paths


def _round_centering(round_id: int, image_id: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed + round_id * 1009 + image_id * 9176)
    cx = 0.50 + rng.uniform(-0.08, 0.08)
    cy = 0.50 + rng.uniform(-0.08, 0.08)
    return max(0.05, min(0.95, cx)), max(0.05, min(0.95, cy))


def build_round_image(
    source_path: Path,
    output_path: Path,
    *,
    size: tuple[int, int],
    round_id: int,
    image_id: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        centering = _round_centering(round_id, image_id, seed)
        fitted = ImageOps.fit(
            rgb,
            size,
            method=Image.Resampling.LANCZOS,
            centering=centering,
        )

        quality = 92 - (round_id % 4)
        buffer = BytesIO()
        fitted.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        output_path.write_bytes(encoded)

    return sha256_bytes(encoded), {
        "source_path": str(source_path.resolve()),
        "centering": [round(centering[0], 4), round(centering[1], 4)],
        "jpeg_quality": quality,
    }


def build_payload(
    round_dir: Path,
    text: str,
    request_model: str,
    images_per_round: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    image_paths = sorted((round_dir / "images").glob("*.jpg"))
    if len(image_paths) != images_per_round:
        raise RuntimeError(
            f"{round_dir} expected {images_per_round} images, found {len(image_paths)}"
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path.resolve().as_uri()},
            }
        )

    return {
        "model": request_model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    source_dir = output_dir / "source_pool"
    if args.source_urls_file is not None:
        source_catalog = fetch_explicit_sources(
            args.source_urls_file.resolve(),
            args.images_per_round,
        )
    else:
        source_catalog = fetch_source_catalog(args.source_list_url, args.images_per_round)
    source_paths = download_source_images(source_dir, source_catalog)

    seen_texts: set[str] = set()
    seen_image_hashes: dict[str, str] = {}
    round_summaries: list[dict[str, Any]] = []

    for round_id in range(args.rounds):
        round_dir = output_dir / f"round_{round_id}"
        image_dir = round_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        text, token_count = build_text(tokenizer, round_id, args.target_text_tokens)
        text_lines = validate_text_uniqueness(text, round_id)
        if text in seen_texts:
            raise RuntimeError(f"round_{round_id} duplicated a previous round text")
        seen_texts.add(text)

        image_hashes: list[str] = []
        image_metadata: list[dict[str, Any]] = []
        for image_id, source_path in enumerate(source_paths):
            output_path = image_dir / f"img_{image_id:02d}.jpg"
            image_hash, transform_meta = build_round_image(
                source_path,
                output_path,
                size=(args.width, args.height),
                round_id=round_id,
                image_id=image_id,
                seed=args.seed,
            )
            if image_hash in seen_image_hashes:
                raise RuntimeError(
                    "duplicate image bytes detected between "
                    f"{seen_image_hashes[image_hash]} and {output_path.resolve()}"
                )
            seen_image_hashes[image_hash] = str(output_path.resolve())
            image_hashes.append(image_hash)
            image_metadata.append(
                {
                    "output_path": str(output_path.resolve()),
                    "sha256": image_hash,
                    **transform_meta,
                }
            )

        payload = build_payload(
            round_dir,
            text,
            args.request_model,
            args.images_per_round,
            args.max_completion_tokens,
        )
        payload_path = round_dir / "payload.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        round_summaries.append(
            {
                "round": round_id,
                "text_tokens": token_count,
                "text_line_count": len(text_lines),
                "image_count": args.images_per_round,
                "payload_path": str(payload_path.resolve()),
                "images": image_metadata,
            }
        )
        print(
            f"round_{round_id}: text_tokens={token_count} "
            f"images={args.images_per_round} payload={payload_path.resolve()}"
        )

    manifest = {
        "tokenizer": args.tokenizer,
        "request_model": args.request_model,
        "output_dir": str(output_dir),
        "rounds": args.rounds,
        "warmup_rounds": list(range(args.warmup_rounds)),
        "test_rounds": list(range(args.warmup_rounds, args.rounds)),
        "target_text_tokens": args.target_text_tokens,
        "images_per_round": args.images_per_round,
        "image_size": {"width": args.width, "height": args.height},
        "max_completion_tokens": args.max_completion_tokens,
        "seed": args.seed,
        "source_list_url": args.source_list_url,
        "source_pool": [
            {
                "photo_id": source.photo_id,
                "author": source.author,
                "width": source.width,
                "height": source.height,
                "download_url": source.download_url,
                "downloaded_path": str(path.resolve()),
                "downloaded_sha256": sha256_file(path),
            }
            for source, path in zip(source_catalog, source_paths, strict=True)
        ],
        "unique_text_count": len(seen_texts),
        "unique_image_count": len(seen_image_hashes),
        "round_summaries": round_summaries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
