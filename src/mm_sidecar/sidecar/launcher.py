from __future__ import annotations

import signal
import threading
import time

from .service import SidecarServiceProcess, sidecar_service_config_from_env


def main() -> None:
    config = sidecar_service_config_from_env()
    service = SidecarServiceProcess(config)
    service.start()

    stop_event = threading.Event()

    def _handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        service.terminate()


if __name__ == "__main__":
    main()
