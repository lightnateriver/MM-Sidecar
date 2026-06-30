from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

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
    def test_materialize_and_load_torch_bf16_local_file_payload(self) -> None:
        descriptor, payload = _make_descriptor_and_payload()
        expected_bf16 = torch.from_numpy(payload.pixel_values).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "MM_SIDECAR_PAYLOAD_DIR": tmpdir,
                "MM_SIDECAR_PAYLOAD_FILE_FORMAT": "torch",
                "MM_SIDECAR_PAYLOAD_DTYPE": "bf16",
            },
            clear=False,
        ):
            stored_descriptor, stored_payload, store_details = (
                materialize_payload_to_local_file(
                    cache_key="cache-torch-bf16",
                    epoch=2,
                    descriptor=descriptor,
                    payload=payload,
                )
            )

            self.assertEqual(stored_descriptor.storage_kind, StorageKind.LOCAL_FILE)
            self.assertEqual(stored_descriptor.payload_dtype, "bfloat16")
            self.assertEqual(stored_payload.payload_dtype, "bfloat16")
            self.assertEqual(store_details["payload_file_format"], "torch")
            self.assertEqual(store_details["payload_stored_dtype"], "bfloat16")
            self.assertGreaterEqual(store_details["payload_tensor_cast_ms"], 0.0)
            self.assertGreaterEqual(store_details["payload_torch_save_ms"], 0.0)
            ref = stored_payload.pixel_values
            self.assertIsInstance(ref, LocalFileTensorPayloadRef)
            assert isinstance(ref, LocalFileTensorPayloadRef)
            self.assertEqual(ref.format, "torch")
            self.assertEqual(ref.dtype, "bfloat16")
            self.assertEqual(ref.nbytes, expected_bf16.numel() * expected_bf16.element_size())
            self.assertEqual(Path(ref.path).suffix, ".torch")

            tensor = load_local_file_tensor_ref(ref)
            self.assertIsInstance(tensor, torch.Tensor)
            assert isinstance(tensor, torch.Tensor)
            self.assertEqual(tensor.dtype, torch.bfloat16)
            self.assertEqual(tuple(tensor.shape), (3, 4))
            self.assertEqual(tensor.numel() * tensor.element_size(), ref.nbytes)
            self.assertTrue(torch.equal(tensor, expected_bf16))

    def test_materialize_and_load_numpy_bf16_local_file_payload(self) -> None:
        descriptor, payload = _make_descriptor_and_payload()
        expected_bf16 = torch.from_numpy(payload.pixel_values).to(torch.bfloat16)
        expected_uint16 = expected_bf16.view(torch.uint16).cpu().numpy()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "MM_SIDECAR_PAYLOAD_DIR": tmpdir,
                "MM_SIDECAR_PAYLOAD_FILE_FORMAT": "numpy_bf16",
                "MM_SIDECAR_PAYLOAD_DTYPE": "bf16",
            },
            clear=False,
        ):
            stored_descriptor, stored_payload, store_details = (
                materialize_payload_to_local_file(
                    cache_key="cache-numpy-bf16",
                    epoch=3,
                    descriptor=descriptor,
                    payload=payload,
                )
            )

            self.assertEqual(stored_descriptor.storage_kind, StorageKind.LOCAL_FILE)
            self.assertEqual(stored_descriptor.payload_dtype, "bfloat16")
            self.assertEqual(stored_payload.payload_dtype, "bfloat16")
            self.assertEqual(store_details["payload_file_format"], "numpy_bf16")
            self.assertEqual(store_details["payload_stored_dtype"], "uint16")
            self.assertGreaterEqual(store_details["payload_tensor_cast_ms"], 0.0)
            self.assertGreaterEqual(store_details["payload_numpy_bf16_save_ms"], 0.0)
            ref = stored_payload.pixel_values
            self.assertIsInstance(ref, LocalFileTensorPayloadRef)
            assert isinstance(ref, LocalFileTensorPayloadRef)
            self.assertEqual(ref.format, "numpy_bf16")
            self.assertEqual(ref.dtype, "bfloat16")
            self.assertEqual(ref.nbytes, int(expected_uint16.nbytes))
            self.assertEqual(Path(ref.path).suffix, ".numpy_bf16")

            array = load_local_file_tensor_ref(ref)
            self.assertEqual(str(array.dtype), "uint16")
            self.assertEqual(tuple(array.shape), (3, 4))
            self.assertEqual(int(array.nbytes), ref.nbytes)
            np.testing.assert_array_equal(array, expected_uint16)
            restored = torch.from_numpy(np.asarray(array)).view(torch.bfloat16)
            self.assertTrue(torch.equal(restored, expected_bf16))

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
            stored_descriptor, stored_payload, store_details = (
                materialize_payload_to_local_file(
                    cache_key="cache-raw",
                    epoch=1,
                    descriptor=descriptor,
                    payload=payload,
                )
            )

            self.assertGreaterEqual(store_details["payload_local_file_write_ms"], 0.0)
            self.assertEqual(stored_descriptor.storage_kind, StorageKind.LOCAL_FILE)
            self.assertEqual(stored_payload.storage_kind, StorageKind.LOCAL_FILE)
            ref = stored_payload.pixel_values
            self.assertIsInstance(ref, LocalFileTensorPayloadRef)
            assert isinstance(ref, LocalFileTensorPayloadRef)
            self.assertEqual(ref.format, "raw")
            self.assertEqual(ref.shape, (3, 4))
            self.assertEqual(ref.dtype, "float32")
            self.assertEqual(ref.nbytes, int(payload.pixel_values.nbytes))
            self.assertEqual(store_details["payload_file_format"], "raw")
            self.assertEqual(store_details["payload_stored_dtype"], "float32")
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
