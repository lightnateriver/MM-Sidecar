from __future__ import annotations

import os
import unittest

from mm_sidecar.sidecar import (
    SidecarServiceConfig,
    connect_sidecar_client_from_env,
    describe_sidecar_service_config,
    sidecar_service_config_from_env,
)


class SidecarServiceConfigTests(unittest.TestCase):
    def test_describe_sidecar_service_config_includes_transport_fields(self) -> None:
        config = SidecarServiceConfig(
            transport="tcp",
            tcp_host="127.0.0.1",
            tcp_port=9911,
            worker_pool_mode="inline",
            start_method="fork",
        )

        payload = describe_sidecar_service_config(config)

        self.assertEqual(payload["transport"], "tcp")
        self.assertEqual(payload["tcp_host"], "127.0.0.1")
        self.assertEqual(payload["tcp_port"], 9911)
        self.assertEqual(payload["worker_pool_mode"], "inline")
        self.assertEqual(payload["start_method"], "fork")

    def test_sidecar_service_config_from_env_is_optional(self) -> None:
        old_env = {
            key: os.environ.get(key)
            for key in ("MM_SIDECAR_TRANSPORT", "MM_SIDECAR_SOCKET_PATH", "MM_SIDECAR_TCP_PORT")
        }
        try:
            for key in old_env:
                os.environ.pop(key, None)

            self.assertIsNone(sidecar_service_config_from_env(required=False))
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_connect_sidecar_client_from_env_without_config_returns_none(self) -> None:
        old_env = {
            key: os.environ.get(key)
            for key in ("MM_SIDECAR_TRANSPORT", "MM_SIDECAR_SOCKET_PATH", "MM_SIDECAR_TCP_PORT")
        }
        try:
            for key in old_env:
                os.environ.pop(key, None)

            self.assertIsNone(connect_sidecar_client_from_env(required=False))
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_sidecar_service_config_from_env_applies_worker_env(self) -> None:
        keys = (
            "MM_SIDECAR_TRANSPORT",
            "MM_SIDECAR_SOCKET_PATH",
            "MM_SIDECAR_TCP_PORT",
            "MM_SIDECAR_WORKER_COUNT",
            "MM_SIDECAR_REUSABLE_CACHE_BYTES",
            "MM_SIDECAR_REUSABLE_TTL_S",
            "MM_SIDECAR_WORKER_START_METHOD",
        )
        old_env = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["MM_SIDECAR_TRANSPORT"] = "unix"
            os.environ["MM_SIDECAR_SOCKET_PATH"] = "/tmp/mm-sidecar-test.sock"
            os.environ["MM_SIDECAR_WORKER_COUNT"] = "3"
            os.environ["MM_SIDECAR_REUSABLE_CACHE_BYTES"] = "123456"
            os.environ["MM_SIDECAR_REUSABLE_TTL_S"] = "42.5"
            os.environ["MM_SIDECAR_WORKER_START_METHOD"] = "fork"

            config = sidecar_service_config_from_env(required=True)
            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.manager.workers.worker_count, 3)
            self.assertEqual(len(config.manager.workers.cpu_affinity_map or ()), 3)
            self.assertEqual(config.manager.cache.max_reusable_bytes, 123456)
            self.assertEqual(config.manager.cache.reusable_entry_ttl_s, 42.5)

            payload = describe_sidecar_service_config(config)
            self.assertEqual(payload["worker_count"], 3)
            self.assertEqual(len(payload["cpu_affinity_map"]), 3)
            self.assertEqual(payload["reusable_cache_bytes"], 123456)
            self.assertEqual(payload["reusable_ttl_s"], 42.5)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
