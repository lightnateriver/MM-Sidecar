from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import uvloop


_REQUEST_CAPTURE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "mm_sidecar_baseline_profile_capture",
    default=None,
)
_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False


def _now_ms() -> float:
    return time.time() * 1000.0


def _perf_ms() -> float:
    return time.perf_counter() * 1000.0


def _debug_route_enabled() -> bool:
    value = os.getenv("MM_BASELINE_PROFILE_ENABLE_DEBUG_ROUTE", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _record_event(label: str, started_at_ms: float, ended_at_ms: float) -> None:
    capture = _REQUEST_CAPTURE.get()
    if capture is None:
        return
    events = capture.setdefault("events", [])
    events.append(
        {
            "label": label,
            "start_ms": started_at_ms,
            "end_ms": ended_at_ms,
            "duration_ms": ended_at_ms - started_at_ms,
        }
    )


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        row = grouped.setdefault(
            event["label"],
            {
                "label": event["label"],
                "calls": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "first_start_ms": event["start_ms"],
                "last_end_ms": event["end_ms"],
            },
        )
        row["calls"] += 1
        row["total_ms"] += event["duration_ms"]
        row["max_ms"] = max(row["max_ms"], event["duration_ms"])
        row["first_start_ms"] = min(row["first_start_ms"], event["start_ms"])
        row["last_end_ms"] = max(row["last_end_ms"], event["end_ms"])

    for row in grouped.values():
        row["avg_ms"] = row["total_ms"] / row["calls"] if row["calls"] else 0.0
        row["wall_span_ms"] = row["last_end_ms"] - row["first_start_ms"]
    return {
        "functions": sorted(
            grouped.values(),
            key=lambda row: row["total_ms"],
            reverse=True,
        ),
        "ordered_functions": sorted(
            grouped.values(),
            key=lambda row: row["first_start_ms"],
        ),
    }


class _BaselineCaptureMiddleware:
    def __init__(self, app: Any, *, app_state: Any) -> None:
        self.app = app
        self.app_state = app_state

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[Any]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        if path.startswith("/mm_sidecar/debug/"):
            await self.app(scope, receive, send)
            return

        request_id = None
        for key, value in scope.get("headers", []):
            if key == b"x-request-id":
                request_id = value.decode("latin-1")
                break
        request_id = request_id or f"baseline-{int(_now_ms())}"

        capture = {
            "request_id": request_id,
            "path": path,
            "method": str(scope.get("method", "UNKNOWN")),
            "started_at_ms": _now_ms(),
            "events": [],
            "timings_ms": {},
        }
        token = _REQUEST_CAPTURE.set(capture)
        response_started = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
                capture["finished_at_ms"] = _now_ms()
                capture["status_code"] = message.get("status")
                capture["duration_ms"] = capture["finished_at_ms"] - capture["started_at_ms"]
                capture.update(_summarize_events(capture["events"]))
                self.app_state.mm_sidecar_last_capture = capture
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            capture["error"] = f"{exc.__class__.__name__}: {exc}"
            capture["finished_at_ms"] = _now_ms()
            capture["status_code"] = 500
            capture["duration_ms"] = capture["finished_at_ms"] - capture["started_at_ms"]
            capture.update(_summarize_events(capture["events"]))
            self.app_state.mm_sidecar_last_capture = capture
            raise
        finally:
            if not response_started:
                capture["finished_at_ms"] = _now_ms()
                capture["duration_ms"] = capture["finished_at_ms"] - capture["started_at_ms"]
                capture.update(_summarize_events(capture["events"]))
                self.app_state.mm_sidecar_last_capture = capture
            _REQUEST_CAPTURE.reset(token)


def _install_request_capture_middleware(app: Any) -> None:
    if getattr(app.state, "_mm_baseline_capture_installed", False):
        return

    app.state._mm_baseline_capture_installed = True
    app.state.mm_sidecar_last_capture = None
    app.state.mm_sidecar_patch = {
        "enabled": True,
        "capture_mode": "baseline_api_server_profiler",
        "debug_route_enabled": _debug_route_enabled(),
    }

    from starlette.middleware import Middleware

    app.user_middleware.insert(
        0,
        Middleware(_BaselineCaptureMiddleware, app_state=app.state),
    )
    app.middleware_stack = app.build_middleware_stack()

    if _debug_route_enabled():
        from fastapi import HTTPException

        async def mm_sidecar_last_capture() -> dict[str, Any]:
            payload = getattr(app.state, "mm_sidecar_last_capture", None)
            if payload is None:
                raise HTTPException(status_code=404, detail="no capture available")
            return payload

        app.router.add_api_route(
            "/mm_sidecar/debug/last_capture",
            mm_sidecar_last_capture,
            methods=["GET"],
        )


def _wrap_async_method(cls: type[Any], attr: str, label: str) -> None:
    original = getattr(cls, attr, None)
    if original is None or getattr(original, "_mm_baseline_profile_wrapped", False):
        return

    @functools.wraps(original)
    async def wrapped(*args: Any, **kwargs: Any):
        started_at_ms = _perf_ms()
        try:
            return await original(*args, **kwargs)
        finally:
            _record_event(label, started_at_ms, _perf_ms())

    wrapped._mm_baseline_profile_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, attr, wrapped)


def _wrap_sync_method(cls: type[Any], attr: str, label: str) -> None:
    original = getattr(cls, attr, None)
    if original is None or getattr(original, "_mm_baseline_profile_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        started_at_ms = _perf_ms()
        try:
            return original(*args, **kwargs)
        finally:
            _record_event(label, started_at_ms, _perf_ms())

    wrapped._mm_baseline_profile_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, attr, wrapped)


def apply_monkey_patches() -> None:
    global _PATCH_APPLIED
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return

        from vllm.entrypoints import chat_utils
        from vllm.entrypoints.openai import api_server
        from vllm.multimodal.media.connector import MediaConnector
        from vllm.multimodal.processing.context import InputProcessingContext
        from vllm.renderers import hf as hf_renderer

        _wrap_async_method(
            chat_utils.AsyncMultiModalContentParser,
            "_image_with_uuid_async",
            "AsyncMultiModalContentParser._image_with_uuid_async",
        )
        _wrap_async_method(
            MediaConnector,
            "load_from_url_async",
            "MediaConnector.load_from_url_async",
        )
        _wrap_async_method(
            hf_renderer.HfRenderer,
            "render_messages_async",
            "HfRenderer.render_messages_async",
        )
        _wrap_sync_method(
            InputProcessingContext,
            "call_hf_processor",
            "InputProcessingContext.call_hf_processor",
        )
        try:
            from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
        except Exception:
            Qwen3VLMultiModalProcessor = None

        if (
            Qwen3VLMultiModalProcessor is not None
            and hasattr(Qwen3VLMultiModalProcessor, "_call_hf_processor")
        ):
            _wrap_sync_method(
                Qwen3VLMultiModalProcessor,
                "_call_hf_processor",
                "Qwen3VLMultiModalProcessor._call_hf_processor",
            )

        original_build_app = api_server.build_app

        @functools.wraps(original_build_app)
        def wrapped_build_app(*args: Any, **kwargs: Any):
            app = original_build_app(*args, **kwargs)
            _install_request_capture_middleware(app)
            return app

        api_server.build_app = wrapped_build_app
        _PATCH_APPLIED = True


def main() -> None:
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    apply_monkey_patches()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server with baseline profiler monkey patch."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
