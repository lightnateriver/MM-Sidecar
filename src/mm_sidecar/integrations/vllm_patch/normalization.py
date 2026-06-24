from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from PIL import Image

from mm_sidecar.contracts import (
    CapturedImageRef,
    MediaSourceRef,
    MediaTransport,
    NormalizedImage,
)
from mm_sidecar.contracts.identity import (
    build_base64_source_key,
    build_http_source_key,
    build_local_source_key,
)


def _decode_base64_size(data_url: str) -> int | None:
    if "," not in data_url:
        return None
    payload = data_url.split(",", 1)[1].strip()
    if not payload:
        return 0
    padding = len(payload) - len(payload.rstrip("="))
    return (len(payload) * 3) // 4 - padding


def extract_data_url_mime_type(data_url: str) -> str | None:
    if not data_url.startswith("data:") or "," not in data_url:
        return None
    header = data_url[5:].split(",", 1)[0]
    mime_type = header.split(";", 1)[0].strip()
    return mime_type or None


def is_http_url(value: str | None) -> bool:
    if not value:
        return False
    scheme = urlsplit(value).scheme.lower()
    return scheme in {"http", "https"}


def is_data_url(value: str | None) -> bool:
    if not value:
        return False
    return value.lower().startswith("data:image/")


def maybe_file_path_from_url(value: str | None) -> str | None:
    if not value:
        return None

    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return url2pathname((parsed.netloc or "") + parsed.path)
    if parsed.scheme == "":
        return value

    return None


def _infer_image_mime_type(image_url: str, image: Image.Image) -> str:
    data_url_mime = extract_data_url_mime_type(image_url)
    if data_url_mime:
        return data_url_mime

    if image.format:
        normalized_format = str(image.format).upper()
        mime_type = Image.MIME.get(normalized_format)
        if mime_type:
            return mime_type

    local_path = maybe_file_path_from_url(image_url)
    if local_path:
        guessed, _ = mimetypes.guess_type(local_path)
        if guessed:
            return guessed

    return "image/unknown"


def build_captured_image_ref(
    *,
    image_url: str,
    media_uuid: str,
    request_scope_key: str,
    item_index: int,
) -> CapturedImageRef:
    if is_data_url(image_url):
        source_ref = MediaSourceRef(
            transport=MediaTransport.BASE64,
            source_key=build_base64_source_key(request_scope_key, item_index),
            media_uuid=media_uuid,
            request_scope_key=request_scope_key,
            image_url=image_url,
            mime_type=extract_data_url_mime_type(image_url),
        )
        byte_size = _decode_base64_size(image_url)
        local_materialized_path = None
        mime_type = extract_data_url_mime_type(image_url)
    elif is_http_url(image_url):
        source_ref = MediaSourceRef(
            transport=MediaTransport.HTTP,
            source_key=build_http_source_key(image_url),
            media_uuid=media_uuid,
            request_scope_key=None,
            image_url=image_url,
        )
        byte_size = None
        local_materialized_path = None
        mime_type = None
    else:
        local_path = maybe_file_path_from_url(image_url)
        if not local_path:
            raise ValueError(f"Unsupported image transport: {image_url!r}")

        path = Path(local_path)
        stat_result = path.stat()
        source_ref = MediaSourceRef(
            transport=MediaTransport.LOCAL_PATH,
            source_key=build_local_source_key(
                str(path),
                mtime_ns=stat_result.st_mtime_ns,
                size_bytes=stat_result.st_size,
            ),
            media_uuid=media_uuid,
            request_scope_key=None,
            local_path=str(path.resolve()),
        )
        byte_size = int(stat_result.st_size)
        local_materialized_path = str(path.resolve())
        guessed, _ = mimetypes.guess_type(str(path.resolve()))
        mime_type = guessed

    return CapturedImageRef(
        source_ref=source_ref,
        mime_type=mime_type,
        byte_size=byte_size,
        local_materialized_path=local_materialized_path,
    )


def build_normalized_image_from_capture(
    *,
    capture: CapturedImageRef,
    image: Image.Image,
) -> NormalizedImage:
    image_url = (
        capture.source_ref.image_url
        or capture.source_ref.local_path
        or ""
    )
    mime_type = capture.mime_type or _infer_image_mime_type(image_url, image)
    orig_size_hw = (int(image.height), int(image.width))
    return NormalizedImage(
        source_ref=capture.source_ref,
        orig_size_hw=orig_size_hw,
        mime_type=mime_type,
        byte_size=capture.byte_size,
        decoded_size_hw=orig_size_hw,
        local_materialized_path=capture.local_materialized_path,
    )


def build_normalized_image_from_url(
    *,
    image_url: str,
    image: Image.Image,
    media_uuid: str,
    request_scope_key: str,
    item_index: int,
) -> NormalizedImage:
    capture = build_captured_image_ref(
        image_url=image_url,
        media_uuid=media_uuid,
        request_scope_key=request_scope_key,
        item_index=item_index,
    )
    return build_normalized_image_from_capture(capture=capture, image=image)
