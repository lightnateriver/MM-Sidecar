from __future__ import annotations

import unittest

from mm_sidecar.contracts import (
    ArtifactDescriptor,
    ImageScheduleItem,
    MediaSourceRef,
    MediaTransport,
    NormalizedImage,
    ProcessorConfig,
    ProcessorSignature,
    ScheduleManifest,
    SidecarContractError,
    StorageKind,
)
from mm_sidecar.contracts.identity import (
    build_base64_source_key,
    build_http_source_key,
    build_local_source_key,
    canonical_http_url,
    infer_transport,
)
from mm_sidecar.contracts.limits import IngressLimits


def _signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="Qwen/Qwen3.5-VL-0.8B",
            revision="local-test",
            processor_name="Qwen2VLImageProcessorFast",
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=65536,
            max_pixels=16777216,
        )
    )


class ContractTests(unittest.TestCase):
    def test_media_source_ref_local_path(self) -> None:
        ref = MediaSourceRef(
            transport=MediaTransport.LOCAL_PATH,
            source_key="local_path:/tmp/a.jpg|1|2",
            media_uuid="uuid-1",
            request_scope_key=None,
            local_path="/tmp/a.jpg",
        )
        self.assertEqual(ref.transport, MediaTransport.LOCAL_PATH)

    def test_media_source_ref_http_requires_url(self) -> None:
        with self.assertRaises(SidecarContractError):
            MediaSourceRef(
                transport=MediaTransport.HTTP,
                source_key="http:https://a/b.jpg",
                media_uuid="uuid-2",
                request_scope_key=None,
            )

    def test_normalized_image_requires_positive_hw(self) -> None:
        ref = MediaSourceRef(
            transport=MediaTransport.BASE64,
            source_key="base64:req:x:image:0",
            media_uuid="uuid-3",
            request_scope_key="req:x",
            image_url="data:image/jpeg;base64,AAAA",
        )
        with self.assertRaises(SidecarContractError):
            NormalizedImage(
                source_ref=ref,
                orig_size_hw=(0, 288),
                mime_type="image/jpeg",
            )

    def test_schedule_manifest_total_must_match(self) -> None:
        item = ImageScheduleItem(
            item_index=0,
            item_identity="item-0",
            processor_signature=_signature(),
            orig_size_hw=(512, 288),
            preprocessed_size_hw=(512, 288),
            image_grid_thw=(1, 32, 18),
            placeholder_token_count=144,
        )
        with self.assertRaises(SidecarContractError):
            ScheduleManifest(
                image_count=1,
                total_placeholder_token_count=145,
                items=(item,),
            )

    def test_artifact_descriptor_valid(self) -> None:
        artifact = ArtifactDescriptor(
            artifact_id="artifact-1",
            item_identity="item-0",
            processor_signature=_signature(),
            image_grid_thw=(1, 32, 18),
            payload_shape=(576, 1536),
            payload_dtype="float32",
            storage_kind=StorageKind.CPU_MEMORY,
            payload_nbytes=576 * 1536 * 4,
        )
        self.assertEqual(artifact.storage_kind, StorageKind.CPU_MEMORY)

    def test_identity_helpers(self) -> None:
        self.assertEqual(
            canonical_http_url("https://example.com:8443/a.jpg?x=1"),
            "https://example.com:8443/a.jpg?x=1",
        )
        self.assertTrue(build_http_source_key("https://example.com/a.jpg").startswith("http:"))
        self.assertTrue(build_local_source_key("/tmp/a.jpg", mtime_ns=1, size_bytes=2).startswith("local_path:"))
        self.assertEqual(build_base64_source_key("req:abc", 3), "base64:req:abc:image:3")

    def test_infer_transport(self) -> None:
        self.assertEqual(
            infer_transport("https://example.com/a.jpg", None),
            MediaTransport.HTTP,
        )
        self.assertEqual(
            infer_transport("data:image/jpeg;base64,AAAA", None),
            MediaTransport.BASE64,
        )
        self.assertEqual(
            infer_transport(None, "/tmp/a.jpg"),
            MediaTransport.LOCAL_PATH,
        )

    def test_limits(self) -> None:
        limits = IngressLimits(
            max_image_count=40,
            max_encoded_bytes=8 * 1024 * 1024,
            max_decoded_bytes=6 * 1024 * 1024,
            max_pixels_per_image=4096 * 4096,
        )
        limits.validate_image_count(40)
        with self.assertRaises(SidecarContractError):
            limits.validate_image_count(41)


if __name__ == "__main__":
    unittest.main()

