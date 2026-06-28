from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from mm_sidecar.integrations.vllm_patch.patches import (
    _RequestCaptureMiddleware,
    _descriptor_only_dummy_allowed,
)


class _FakeApp:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        self.calls.append(str(scope.get("path")))
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )


async def _receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def _make_send_collector(messages: list[dict[str, Any]]):
    async def _send(message: dict[str, Any]) -> None:
        messages.append(message)

    return _send


class VllmPatchMiddlewareTests(unittest.TestCase):
    def test_descriptor_only_dummy_waits_until_min_image_count(self) -> None:
        with patch.dict(
            "os.environ",
            {"MM_SIDECAR_MIN_IMAGE_COUNT": "2"},
            clear=False,
        ):
            self.assertFalse(_descriptor_only_dummy_allowed(0))
            self.assertTrue(_descriptor_only_dummy_allowed(1))

        with patch.dict(
            "os.environ",
            {"MM_SIDECAR_MIN_IMAGE_COUNT": "1"},
            clear=False,
        ):
            self.assertTrue(_descriptor_only_dummy_allowed(0))

    def test_debug_route_bypasses_capture(self) -> None:
        app = _FakeApp()
        state = SimpleNamespace(mm_sidecar_last_capture={"request_id": "keep"})
        middleware = _RequestCaptureMiddleware(app, app_state=state)
        sent: list[dict[str, Any]] = []

        asyncio.run(
            middleware(
                {"type": "http", "method": "GET", "path": "/mm_sidecar/debug/last_capture", "headers": []},
                _receive,
                _make_send_collector(sent),
            )
        )

        self.assertEqual(app.calls, ["/mm_sidecar/debug/last_capture"])
        self.assertEqual(state.mm_sidecar_last_capture, {"request_id": "keep"})
        self.assertEqual(sent[0]["status"], 200)

    def test_regular_route_updates_last_capture(self) -> None:
        app = _FakeApp()
        state = SimpleNamespace(mm_sidecar_last_capture=None, mm_sidecar_client=None)
        middleware = _RequestCaptureMiddleware(app, app_state=state)
        sent: list[dict[str, Any]] = []

        asyncio.run(
            middleware(
                {"type": "http", "method": "GET", "path": "/v1/models", "headers": []},
                _receive,
                _make_send_collector(sent),
            )
        )

        self.assertEqual(app.calls, ["/v1/models"])
        self.assertIsNotNone(state.mm_sidecar_last_capture)
        self.assertIsNotNone(state.mm_sidecar_last_capture_obj)
        assert state.mm_sidecar_last_capture is not None
        self.assertEqual(state.mm_sidecar_last_capture["path"], "/v1/models")
        self.assertEqual(state.mm_sidecar_last_capture["status_code"], 200)
        self.assertEqual(sent[0]["status"], 200)

    @patch("mm_sidecar.integrations.vllm_patch.patches.connect_sidecar_client_from_env")
    @patch("mm_sidecar.integrations.vllm_patch.patches.describe_sidecar_service_config")
    @patch("mm_sidecar.integrations.vllm_patch.patches.sidecar_service_config_from_env")
    @patch("mm_sidecar.integrations.vllm_patch.patches.describe_sidecar_runtime_config")
    def test_patch_initialization_uses_sidecar_client(
        self,
        describe_runtime_config,
        sidecar_service_config_from_env,
        describe_service_config,
        connect_sidecar_client_from_env,
    ) -> None:
        from mm_sidecar.integrations.vllm_patch import patches

        app = _FakeApp()
        app.state = SimpleNamespace()
        app.add_event_handler = unittest.mock.Mock()
        app.user_middleware = []
        app.build_middleware_stack = unittest.mock.Mock(return_value=None)

        describe_runtime_config.return_value = {"worker_count": 1}
        sidecar_service_config = SimpleNamespace()
        sidecar_service_config_from_env.return_value = sidecar_service_config
        describe_service_config.return_value = {"transport": "tcp"}
        connect_sidecar_client_from_env.return_value = SimpleNamespace(close=lambda: None)

        with patch.object(patches, "_debug_route_enabled", return_value=False):
            patches._install_request_capture_middleware(app)

        self.assertIsNotNone(getattr(app.state, "mm_sidecar_client", None))
        self.assertNotIn("mm_sidecar_manager", app.state.__dict__)
        self.assertEqual(app.state.mm_sidecar_patch["capture_mode"], "api_server_sidecar_client")

    @patch("mm_sidecar.integrations.vllm_patch.patches.connect_sidecar_client_from_env")
    @patch("mm_sidecar.integrations.vllm_patch.patches.describe_sidecar_service_config")
    @patch("mm_sidecar.integrations.vllm_patch.patches.sidecar_service_config_from_env")
    @patch("mm_sidecar.integrations.vllm_patch.patches.describe_sidecar_runtime_config")
    def test_patch_initialization_without_sidecar_is_allowed(
        self,
        describe_runtime_config,
        sidecar_service_config_from_env,
        describe_service_config,
        connect_sidecar_client_from_env,
    ) -> None:
        from mm_sidecar.integrations.vllm_patch import patches

        app = _FakeApp()
        app.state = SimpleNamespace()
        app.add_event_handler = unittest.mock.Mock()
        app.user_middleware = []
        app.build_middleware_stack = unittest.mock.Mock(return_value=None)

        describe_runtime_config.return_value = {"worker_count": 1}
        sidecar_service_config_from_env.return_value = None
        describe_service_config.return_value = None
        connect_sidecar_client_from_env.return_value = None

        with patch.object(patches, "_debug_route_enabled", return_value=False):
            patches._install_request_capture_middleware(app)

        self.assertIsNone(getattr(app.state, "mm_sidecar_client", None))
        self.assertNotIn("mm_sidecar_manager", app.state.__dict__)
        self.assertEqual(app.state.mm_sidecar_patch["capture_mode"], "api_server_no_sidecar")


if __name__ == "__main__":
    unittest.main()
