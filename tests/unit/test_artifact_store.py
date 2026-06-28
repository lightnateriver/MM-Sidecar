from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mm_sidecar.contracts import (
    ArtifactDescriptor,
    ImageTensorPayload,
    ProcessorConfig,
    ProcessorSignature,
    StorageKind,
)
from mm_sidecar.contracts.media_source import LocalFileTensorPayloadRef
from mm_sidecar.sidecar.artifact_store import (
    load_local_file_tensor_ref,
    materialize_payload_to_local_file,
)


def _make_signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="qwen3.5-vl",
            revision="artifact-store-test",
            processor_name="qwen-basic",
            patch_size=14,
            merge_size=2,
            temporal_patch_size=1,
            min_pixels=4,
            max_pixels=288 * 512,
        )
    )


def _make_descriptor_and_payload() -> tuple[ArtifactDescriptor, ImageTensorPayload]:
    pixel_values = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptor = ArtifactDescriptor(
        artifact_id="artifact-raw",
        item_identity="image-raw",
        processor_signature=_make_signature(),
        image_grid_thw=(1, 3, 4),
        payload_shape=(3, 4),
        payload_dtype="float32",
        storage_kind=StorageKind.CPU_MEMORY,
        payload_nbytes=int(pixel_values.nbytes),
    )
    payload = ImageTensorPayload(
        pixel_values=pixel_values,
        image_grid_thw=(1, 3, 4),
        payload_shape=(3, 4),
        payload_dtype="float32",
        storage_kind=StorageKind.CPU_MEMORY,
    )
    return descriptor, payload


class ArtifactStoreTests(unittest.TestCase):
    def test_materialize_and_load_raw_local_file_payload(self) -> None:
        descriptor, payload = _make_descriptor_and_payload()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "MM_SIDECAR_PAYLOAD_DIR": tmpdir,
                "MM_SIDECAR_PAYLOAD_FILE_FORMAT": "raw",
            },
            clear=False,
        ):
            stored_descriptor, stored_payload, write_ms = (
                materialize_payload_to_local_file(
                    cache_key="cache-raw",
                    epoch=1,
                    descriptor=descriptor,
                    payload=payload,
                )
            )

            self.assertGreaterEqual(write_ms, 0.0)
            self.assertEqual(stored_descriptor.storage_kind, StorageKind.LOCAL_FILE)
            self.assertEqual(stored_payload.storage_kind, StorageKind.LOCAL_FILE)
            ref = stored_payload.pixel_values
            self.assertIsInstance(ref, LocalFileTensorPayloadRef)
            assert isinstance(ref, LocalFileTensorPayloadRef)
            self.assertEqual(ref.format, "raw")
            self.assertEqual(ref.shape, (3, 4))
            self.assertEqual(ref.dtype, "float32")
            self.assertEqual(ref.nbytes, int(payload.pixel_values.nbytes))
            self.assertEqual(Path(ref.path).suffix, ".raw")
            self.assertEqual(Path(ref.path).stat().st_size, ref.nbytes)

            mmap_array = load_local_file_tensor_ref(ref)
            self.assertIsInstance(mmap_array, np.memmap)
            np.testing.assert_array_equal(mmap_array, payload.pixel_values)
            del mmap_array

            with mock.patch.dict(
                os.environ,
                {"MM_SIDECAR_PAYLOAD_MMAP": "0"},
                clear=False,
            ):
                array = load_local_file_tensor_ref(ref)
            self.assertNotIsInstance(array, np.memmap)
            np.testing.assert_array_equal(array, payload.pixel_values)


if __name__ == "__main__":
    unittest.main()
