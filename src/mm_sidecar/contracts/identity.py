from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from .enums import MediaTransport
from .errors import SidecarContractError
from .enums import SidecarErrorCode


def build_local_source_key(path: str, *, mtime_ns: int, size_bytes: int) -> str:
    resolved = str(Path(path).resolve())
    return f"local_path:{resolved}|{mtime_ns}|{size_bytes}"


def build_http_source_key(url: str) -> str:
    canonical = canonical_http_url(url)
    if canonical is None:
        raise SidecarContractError(
            SidecarErrorCode.INVALID_SOURCE,
            "http source key requires a canonical http/https url",
        )
    return f"http:{canonical}"


def build_base64_source_key(request_scope_key: str, item_index: int) -> str:
    return f"base64:{request_scope_key}:image:{item_index}"


def canonical_http_url(url: str | None) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    port = f":{parsed.port}" if parsed.port is not None else ""
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}{path}{query}"


def infer_transport(image_url: str | None, local_path: str | None) -> MediaTransport:
    if local_path:
        return MediaTransport.LOCAL_PATH
    if isinstance(image_url, str) and image_url.lower().startswith("data:image/"):
        return MediaTransport.BASE64
    if canonical_http_url(image_url) is not None:
        return MediaTransport.HTTP
    raise SidecarContractError(
        SidecarErrorCode.INVALID_TRANSPORT,
        "could not infer transport from image_url/local_path",
    )

