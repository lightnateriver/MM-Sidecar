from __future__ import annotations

import os

import uvloop

from mm_sidecar.integrations.vllm_patch.patches import apply_monkey_patches


def main() -> None:
    # Child EngineCore/worker processes start fresh interpreters in some vLLM
    # configurations. sitecustomize uses this flag to install the same external
    # monkey patches in those children without modifying installed vLLM files.
    os.environ.setdefault("MM_SIDECAR_AUTO_PATCH", "1")
    # In the patched launcher, the intended fast path is for the API server to
    # capture only image descriptors and let sidecar/worker-side replacement
    # provide the real tensors later. Callers can still opt out with
    # MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE=0.
    os.environ.setdefault("MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE", "1")

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
        description="vLLM OpenAI-Compatible RESTful API server with mm-sidecar monkey patch."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
