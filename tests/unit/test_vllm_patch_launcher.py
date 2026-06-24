from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType
from unittest import mock


class VllmPatchLauncherTests(unittest.TestCase):
    def test_launcher_defaults_descriptor_only_capture_on(self) -> None:
        fake_uvloop = mock.Mock()
        fake_cli_args = ModuleType("vllm.entrypoints.openai.cli_args")
        fake_cli_args.make_arg_parser = lambda parser: parser
        fake_cli_args.validate_parsed_serve_args = lambda args: None
        fake_api_server = ModuleType("vllm.entrypoints.openai.api_server")
        fake_api_server.run_server = mock.Mock()
        fake_utils = ModuleType("vllm.entrypoints.utils")
        fake_utils.cli_env_setup = mock.Mock()
        fake_argparse_utils = ModuleType("vllm.utils.argparse_utils")

        class _FakeParser:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def parse_args(self):
                return object()

        fake_argparse_utils.FlexibleArgumentParser = _FakeParser
        fake_vllm = ModuleType("vllm")
        fake_entrypoints = ModuleType("vllm.entrypoints")
        fake_openai = ModuleType("vllm.entrypoints.openai")
        fake_utils_pkg = ModuleType("vllm.utils")

        with mock.patch.dict(
            sys.modules,
            {
                "uvloop": fake_uvloop,
                "vllm": fake_vllm,
                "vllm.entrypoints": fake_entrypoints,
                "vllm.entrypoints.openai": fake_openai,
                "vllm.entrypoints.openai.api_server": fake_api_server,
                "vllm.entrypoints.openai.cli_args": fake_cli_args,
                "vllm.entrypoints.utils": fake_utils,
                "vllm.utils": fake_utils_pkg,
                "vllm.utils.argparse_utils": fake_argparse_utils,
            },
        ), mock.patch.dict(os.environ, {}, clear=True):
            from mm_sidecar.integrations.vllm_patch import launcher

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.launcher.apply_monkey_patches"
            ):
                launcher.main()

            self.assertEqual(os.environ["MM_SIDECAR_AUTO_PATCH"], "1")
            self.assertEqual(os.environ["MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE"], "1")


if __name__ == "__main__":
    unittest.main()
