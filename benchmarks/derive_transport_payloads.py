#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive http/base64/local payload directories from an existing "
            "local-path multimodal dataset."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--transport",
        choices=("local_path", "http", "base64"),
        required=True,
    )
    parser.add_argument("--http-base-url")
    return parser.parse_args()


def _relative_media_path(path: Path, source_dir: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(source_dir.resolve())
    except ValueError:
        relative_path = Path(path.name)
    return relative_path.as_posix()


def _rewrite_content_item(
    item: dict[str, Any],
    *,
    transport: str,
    http_base_url: str | None,
    source_dir: Path,
) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type != "image_url":
        return item

    image_url = item.get("image_url") or {}
    url = image_url.get("url")
    if not isinstance(url, str):
        return item

    path = Path(url.removeprefix("file://"))
    if transport == "local_path":
        rewritten_url = path.resolve().as_uri()
    elif transport == "http":
        if not http_base_url:
            raise ValueError("--http-base-url is required for http transport")
        rewritten_url = http_base_url.rstrip("/") + "/" + _relative_media_path(
            path, source_dir
        )
    else:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        rewritten_url = f"data:image/jpeg;base64,{payload}"

    new_item = dict(item)
    new_image_url = dict(image_url)
    new_image_url["url"] = rewritten_url
    new_item["image_url"] = new_image_url
    return new_item


def _rewrite_payload(
    payload: dict[str, Any],
    *,
    transport: str,
    http_base_url: str | None,
    source_dir: Path,
) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(payload))
    messages = rewritten.get("messages") or []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        message["content"] = [
            _rewrite_content_item(
                item,
                transport=transport,
                http_base_url=http_base_url,
                source_dir=source_dir,
            )
            if isinstance(item, dict)
            else item
            for item in content
        ]
    return rewritten


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = source_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    new_manifest = json.loads(json.dumps(manifest))
    new_manifest["output_dir"] = str(output_dir)
    new_manifest["transport"] = args.transport
    if args.http_base_url:
        new_manifest["http_base_url"] = args.http_base_url

    for round_summary in new_manifest.get("round_summaries", []):
        round_id = int(round_summary["round"])
        source_payload_path = source_dir / f"round_{round_id}" / "payload.json"
        target_round_dir = output_dir / f"round_{round_id}"
        target_round_dir.mkdir(parents=True, exist_ok=True)
        target_payload_path = target_round_dir / "payload.json"
        payload = json.loads(source_payload_path.read_text(encoding="utf-8"))
        rewritten = _rewrite_payload(
            payload,
            transport=args.transport,
            http_base_url=args.http_base_url,
            source_dir=source_dir,
        )
        target_payload_path.write_text(
            json.dumps(rewritten, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        round_summary["payload_path"] = str(target_payload_path.resolve())

    out_manifest_path = output_dir / "dataset_manifest.json"
    out_manifest_path.write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"manifest={out_manifest_path}")


if __name__ == "__main__":
    main()
