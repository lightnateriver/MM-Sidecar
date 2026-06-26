from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Awaitable, Callable
from functools import wraps
from types import SimpleNamespace
from typing import Any

from mm_sidecar.integrations.vllm_patch.context import (
    RequestCapture,
    get_current_capture,
    reset_current_capture,
    set_current_capture,
)
from mm_sidecar.integrations.vllm_patch.normalization import (
    build_captured_image_ref,
    build_normalized_image_from_capture,
    probe_normalized_image_from_capture,
)
from mm_sidecar.integrations.vllm_patch.sidecar_bridge import (
    describe_sidecar_runtime_config,
    prepare_capture_for_sidecar,
    prepare_single_capture_item_for_sidecar,
    refresh_capture_for_debug,
)
from mm_sidecar.integrations.vllm_patch.carrier import (
    attach_sidecar_payload_to_params,
)
from mm_sidecar.integrations.vllm_patch.api_fast_path import (
    descriptor_only_capture_enabled,
    try_apply_api_fast_path,
)
from mm_sidecar.integrations.vllm_patch.worker_sidecar import (
    install_gpu_model_runner_patch,
)
from mm_sidecar.sidecar import (
    connect_sidecar_client_from_env,
    describe_sidecar_service_config,
    sidecar_service_config_from_env,
)

_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False
_PATCH_STATE: dict[str, Any] = {
    "applied": False,
    "async_image_capture": False,
    "sync_image_capture": False,
    "render_capture": False,
    "api_fast_path": False,
    "debug_route_default": False,
}


class _RequestCaptureMiddleware:
    def __init__(self, app: Any, *, app_state: Any) -> None:
        self.app = app
        self.app_state = app_state

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[Any]]) -> None:
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
        if not request_id:
            request_id = _new_request_id()

        capture = RequestCapture(
            request_id=request_id,
            method=str(scope.get("method", "UNKNOWN")),
            path=path,
            sidecar_manager=getattr(self.app_state, "mm_sidecar_client", None),
        )
        token = set_current_capture(capture)
        response_started = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
                capture.finalize(status_code=message.get("status"))
                refresh_capture_for_debug(capture)
                self.app_state.mm_sidecar_last_capture = capture.to_dict()

                raw_headers = list(message.get("headers") or [])
                raw_headers.append((b"x-mm-sidecar-request-id", request_id.encode("latin-1")))
                raw_headers.append(
                    (
                        b"x-mm-sidecar-image-count",
                        str(capture.reserved_image_count).encode("ascii"),
                    )
                )
                message["headers"] = raw_headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            capture.add_error(f"request failed: {exc.__class__.__name__}: {exc}")
            capture.finalize(status_code=500)
            refresh_capture_for_debug(capture)
            self.app_state.mm_sidecar_last_capture = capture.to_dict()
            raise
        finally:
            if not response_started and capture.finished_at_ms is None:
                capture.finalize(status_code=None)
                refresh_capture_for_debug(capture)
                self.app_state.mm_sidecar_last_capture = capture.to_dict()
            reset_current_capture(token)


def _new_request_id() -> str:
    return f"mm-sidecar-{uuid.uuid4().hex}"


def _debug_route_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_ENABLE_DEBUG_ROUTE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _capture_image_result(
    *,
    item_index: int | None,
    image_url: str | None,
    image: object | None,
    explicit_uuid: str | None,
) -> None:
    capture = get_current_capture()
    if capture is None or item_index is None or image_url is None or image is None:
        return

    media_uuid = explicit_uuid or f"{capture.request_id}:image:{item_index}"

    try:
        captured_ref = build_captured_image_ref(
            image_url=image_url,
            media_uuid=media_uuid,
            request_scope_key=capture.request_id,
            item_index=item_index,
        )
        normalized_image = build_normalized_image_from_capture(
            capture=captured_ref,
            image=image,
        )
    except Exception as exc:
        capture.add_error(
            f"image[{item_index}] normalization failed: "
            f"{exc.__class__.__name__}: {exc}"
        )
        return

    capture.add_normalized_image(
        item_index=item_index,
        media_uuid=media_uuid,
        normalized_image=normalized_image,
    )


def _capture_image_descriptor(
    *,
    item_index: int | None,
    image_url: str | None,
    explicit_uuid: str | None,
) -> Any | None:
    capture = get_current_capture()
    if capture is None or item_index is None or image_url is None:
        return None

    media_uuid = explicit_uuid or f"{capture.request_id}:image:{item_index}"
    try:
        image_ref = build_captured_image_ref(
            image_url=image_url,
            media_uuid=media_uuid,
            request_scope_key=capture.request_id,
            item_index=item_index,
        )
    except Exception as exc:
        capture.add_error(
            f"image[{item_index}] descriptor capture failed: "
            f"{exc.__class__.__name__}: {exc}"
        )
        return None

    capture.add_captured_image_ref(
        item_index=item_index,
        media_uuid=media_uuid,
        image_ref=image_ref,
    )
    return image_ref


def _prepare_sidecar_if_possible(
    *,
    item_index: int | None,
    image_url: str | None,
    captured_ref: Any | None,
    renderer: Any | None,
    params: Any | None,
) -> Any | None:
    capture = get_current_capture()
    if (
        capture is None
        or item_index is None
        or image_url is None
        or captured_ref is None
        or renderer is None
        or params is None
    ):
        return None
    try:
        return prepare_single_capture_item_for_sidecar(
            capture,
            renderer,
            params,
            item_index=item_index,
            captured_ref=captured_ref,
        )
    except Exception as exc:
        capture.add_error(
            f"image[{item_index}] sidecar prepare failed: "
            f"{exc.__class__.__name__}: {exc}"
        )
        return None


def _try_capture_descriptor_only_probe(
    *,
    item_index: int | None,
    captured_ref: Any | None,
    explicit_uuid: str | None,
) -> bool:
    capture = get_current_capture()
    if capture is None or item_index is None or captured_ref is None:
        return False

    try:
        normalized_image = probe_normalized_image_from_capture(
            capture=captured_ref,
        )
    except Exception as exc:
        capture.add_error(
            f"image[{item_index}] descriptor-only probe failed: "
            f"{exc.__class__.__name__}: {exc}"
        )
        return False
    if normalized_image is None:
        return False

    media_uuid = explicit_uuid or f"{capture.request_id}:image:{item_index}"
    capture.add_normalized_image(
        item_index=item_index,
        media_uuid=media_uuid,
        normalized_image=normalized_image,
    )
    return True


def _descriptor_only_dummy_image() -> Any:
    from PIL import Image

    image = Image.new("RGB", (1, 1), color=(0, 0, 0))
    setattr(image, "_mm_sidecar_descriptor_only_dummy", True)
    return image


def _install_request_capture_middleware(app: Any) -> None:
    if getattr(app.state, "_mm_sidecar_capture_installed", False):
        return

    app.state._mm_sidecar_capture_installed = True
    app.state.mm_sidecar_last_capture = None
    service_config = sidecar_service_config_from_env(required=False)
    app.state.mm_sidecar_client = connect_sidecar_client_from_env(required=False)
    app.state.mm_sidecar_patch = {
        "enabled": True,
        "debug_route_enabled": _debug_route_enabled(),
        "capture_mode": (
            "api_server_sidecar_client"
            if app.state.mm_sidecar_client is not None
            else "api_server_no_sidecar"
        ),
        "sidecar_runtime": describe_sidecar_runtime_config(),
        "sidecar_service": describe_sidecar_service_config(service_config),
    }
    if app.state.mm_sidecar_client is not None:
        app.add_event_handler("shutdown", app.state.mm_sidecar_client.close)

    try:
        from starlette.middleware import Middleware
    except Exception:
        class Middleware:  # type: ignore[no-redef]
            def __init__(self, cls: Any, *args: Any, **kwargs: Any) -> None:
                self.cls = cls
                self.args = args
                self.kwargs = kwargs

    app.user_middleware.insert(
        0,
        Middleware(_RequestCaptureMiddleware, app_state=app.state),
    )
    app.middleware_stack = app.build_middleware_stack()

    if _debug_route_enabled():
        from fastapi import HTTPException

        async def mm_sidecar_last_capture() -> dict[str, Any]:
            payload = getattr(app.state, "mm_sidecar_last_capture", None)
            if payload is None:
                raise HTTPException(status_code=404, detail="no capture available")
            return payload

        async def mm_sidecar_manager_stats() -> dict[str, Any]:
            if app.state.mm_sidecar_client is None:
                raise HTTPException(status_code=503, detail="sidecar client unavailable")
            stats = app.state.mm_sidecar_client.stats()
            return {
                "queued_items": stats.queued_items,
                "running_items": stats.running_items,
                "ready_items": stats.ready_items,
                "failed_items": stats.failed_items,
                "fallback_claimed_items": stats.fallback_claimed_items,
                "reusable_cache_items": stats.reusable_cache_items,
                "reusable_cache_bytes": stats.reusable_cache_bytes,
                "active_inflight_items": stats.active_inflight_items,
                "observed_at_ms": stats.observed_at_ms,
            }

        app.router.add_api_route(
            "/mm_sidecar/debug/last_capture",
            mm_sidecar_last_capture,
            methods=["GET"],
        )
        app.router.add_api_route(
            "/mm_sidecar/debug/manager_stats",
            mm_sidecar_manager_stats,
            methods=["GET"],
        )


def apply_monkey_patches() -> None:
    global _PATCH_APPLIED

    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return

        from vllm.entrypoints import chat_utils
        from vllm.entrypoints.openai import api_server
        from vllm.multimodal.processing import processor as mm_processor_module
        from vllm.renderers import hf as hf_renderer
        from vllm.v1.worker import gpu_model_runner

        original_async_image = chat_utils.AsyncMultiModalContentParser._image_with_uuid_async
        original_sync_parse_image = chat_utils.MultiModalContentParser.parse_image
        original_build_app = api_server.build_app
        original_render_messages = hf_renderer.HfRenderer.render_messages
        original_render_messages_async = hf_renderer.HfRenderer.render_messages_async
        original_mm_processor_apply = mm_processor_module.BaseMultiModalProcessor.apply

        @wraps(original_async_image)
        async def wrapped_async_image(self: Any, image_url: str | None, uuid: str | None):
            capture = get_current_capture()
            item_index = capture.reserve_image_slot() if capture is not None else None
            captured_ref = _capture_image_descriptor(
                item_index=item_index,
                image_url=image_url,
                explicit_uuid=uuid,
            )
            sidecar_prepare_result = _prepare_sidecar_if_possible(
                item_index=item_index,
                image_url=image_url,
                captured_ref=captured_ref,
                renderer=self,
                params=SimpleNamespace(
                    mm_processor_kwargs=(
                        getattr(self, "_mm_processor_kwargs", None) or {}
                    )
                ),
            )
            if (
                descriptor_only_capture_enabled()
                and sidecar_prepare_result is not None
                and _try_capture_descriptor_only_probe(
                    item_index=item_index,
                    captured_ref=captured_ref,
                    explicit_uuid=uuid,
                )
            ):
                return _descriptor_only_dummy_image(), uuid
            try:
                image, resolved_uuid = await original_async_image(self, image_url, uuid)
            except Exception as exc:
                if capture is not None and item_index is not None:
                    capture.add_error(
                        f"image[{item_index}] fetch failed: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                raise

            _capture_image_result(
                item_index=item_index,
                image_url=image_url,
                image=image,
                explicit_uuid=resolved_uuid or uuid,
            )
            return image, resolved_uuid

        @wraps(original_sync_parse_image)
        def wrapped_sync_parse_image(self: Any, image_url: str | None, uuid: str | None = None) -> None:
            capture = get_current_capture()
            item_index = capture.reserve_image_slot() if capture is not None else None
            captured_ref = _capture_image_descriptor(
                item_index=item_index,
                image_url=image_url,
                explicit_uuid=uuid,
            )
            sidecar_prepare_result = _prepare_sidecar_if_possible(
                item_index=item_index,
                image_url=image_url,
                captured_ref=captured_ref,
                renderer=self,
                params=SimpleNamespace(
                    mm_processor_kwargs=(
                        getattr(self, "_mm_processor_kwargs", None) or {}
                    )
                ),
            )

            if (
                descriptor_only_capture_enabled()
                and sidecar_prepare_result is not None
                and _try_capture_descriptor_only_probe(
                    item_index=item_index,
                    captured_ref=captured_ref,
                    explicit_uuid=uuid,
                )
            ):
                image = _descriptor_only_dummy_image()
            else:
                try:
                    image = self._connector.fetch_image(image_url) if image_url else None
                except Exception as exc:
                    if capture is not None and item_index is not None:
                        capture.add_error(
                            f"image[{item_index}] fetch failed: "
                            f"{exc.__class__.__name__}: {exc}"
                        )
                    raise

                _capture_image_result(
                    item_index=item_index,
                    image_url=image_url,
                    image=image,
                    explicit_uuid=uuid,
                )

            placeholder = self._tracker.add("image", (image, uuid))
            self._add_placeholder("image", placeholder)

        @wraps(original_render_messages)
        def wrapped_render_messages(self: Any, messages: list[Any], params: Any):
            conversation, prompt = original_render_messages(self, messages, params)
            capture = get_current_capture()
            if capture is not None:
                capture.add_render_metadata(prompt)
                try:
                    prepare_capture_for_sidecar(capture, self, params)
                    attach_sidecar_payload_to_params(params, capture)
                except Exception as exc:
                    capture.add_error(
                        f"sidecar prepare failed: {exc.__class__.__name__}: {exc}"
                    )
            return conversation, prompt

        @wraps(original_render_messages_async)
        async def wrapped_render_messages_async(self: Any, messages: list[Any], params: Any):
            conversation, prompt = await original_render_messages_async(self, messages, params)
            capture = get_current_capture()
            if capture is not None:
                capture.add_render_metadata(prompt)
                try:
                    prepare_capture_for_sidecar(capture, self, params)
                    attach_sidecar_payload_to_params(params, capture)
                except Exception as exc:
                    capture.add_error(
                        f"sidecar prepare failed: {exc.__class__.__name__}: {exc}"
                    )
            return conversation, prompt

        @wraps(original_build_app)
        def wrapped_build_app(*args: Any, **kwargs: Any):
            app = original_build_app(*args, **kwargs)
            _install_request_capture_middleware(app)
            return app

        @wraps(original_mm_processor_apply)
        def wrapped_mm_processor_apply(self: Any, inputs: Any, timing_ctx: Any):
            try:
                fast_result = try_apply_api_fast_path(self, inputs, timing_ctx)
            except Exception as exc:
                capture = get_current_capture()
                if capture is not None:
                    capture.add_error(
                        "api fast path failed: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                fast_result = None
            if fast_result is not None:
                return fast_result
            return original_mm_processor_apply(self, inputs, timing_ctx)

        chat_utils.AsyncMultiModalContentParser._image_with_uuid_async = wrapped_async_image
        chat_utils.MultiModalContentParser.parse_image = wrapped_sync_parse_image
        hf_renderer.HfRenderer.render_messages = wrapped_render_messages
        hf_renderer.HfRenderer.render_messages_async = wrapped_render_messages_async
        api_server.build_app = wrapped_build_app
        mm_processor_module.BaseMultiModalProcessor.apply = wrapped_mm_processor_apply
        install_gpu_model_runner_patch(gpu_model_runner.GPUModelRunner)

        _PATCH_STATE.update(
            {
                "applied": True,
                "async_image_capture": True,
                "sync_image_capture": True,
                "render_capture": True,
                "api_fast_path": True,
                "worker_sidecar_patch": True,
                "debug_route_default": _debug_route_enabled(),
            }
        )
        _PATCH_APPLIED = True


def get_patch_state() -> dict[str, Any]:
    return dict(_PATCH_STATE)
