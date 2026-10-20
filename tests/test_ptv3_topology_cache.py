import pytest
import torch

from fpsgen.hier_occ_flow.ptv3_inference_topology import PTv3InferenceTopology
from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point
from fpsgen.voxel_point.vendor.pwm.ptv3_reference.serialization import encode


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_top_level_serialization_matches_point(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    coords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], device=device)
    batch = torch.zeros(4, dtype=torch.long, device=device)
    centers = coords.float()
    feat = torch.randn(4, 5, device=device)
    point = Point({"feat": feat, "coord": centers, "grid_coord": coords.int(),
                   "batch": batch, "sparse_shape": [4, 4, 4]})
    point.serialization(order=["z", "z-trans", "hilbert", "hilbert-trans"], depth=2,
                        shuffle_orders=False)
    cache = PTv3InferenceTopology.build(
        coords, centers, batch, [4, 4, 4], ["z", "z-trans", "hilbert", "hilbert-trans"],
        2, scene_count=1, level="serialization")
    dynamic = feat * 2
    fresh = cache.make_point(dynamic, torch.zeros(4, 3, device=device))
    assert torch.equal(point.serialized_code, fresh.serialized_code)
    assert torch.equal(point.serialized_order, fresh.serialized_order)
    assert torch.equal(point.serialized_inverse, fresh.serialized_inverse)
    assert torch.equal(point.grid_coord, fresh.grid_coord)
    assert torch.equal(point.batch, fresh.batch)
    assert torch.equal(point.offset, fresh.offset)
    assert fresh.feat.data_ptr() == dynamic.data_ptr()


def test_cache_rejects_empty_or_mismatched_geometry():
    coords = torch.zeros((0, 3), dtype=torch.int32)
    with pytest.raises(ValueError):
        PTv3InferenceTopology.build(coords, coords.float(), torch.zeros(0, dtype=torch.long),
                                    [4, 4, 4], ["z"], 2, scene_count=1, level="serialization")


def test_selective_precision_uses_fp32_norm_statistics_without_changing_dtype():
    from fpsgen.hier_occ_flow.stage2_model import single_scene_norm
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointInstanceNorm

    Norm = single_scene_norm(PointInstanceNorm)
    norm = Norm(3, affine=False)
    norm._v3_fp32_stats = True
    feat = torch.tensor([[1.0, 2.0, 5.0], [2.0, 4.0, 8.0],
                         [4.0, 7.0, 10.0]], dtype=torch.float16)
    p = Point({"feat": feat, "batch": torch.zeros(3, dtype=torch.long)})
    actual = norm(p).feat
    centered = feat.float() - feat.float().mean(0, keepdim=True)
    expected = (centered * torch.rsqrt(centered.square().mean(0, keepdim=True) + norm.eps)).half()
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_multilevel_attention_and_pooling_maps_are_reused(monkeypatch):
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import SerializedAttention, SerializedPooling

    # The metadata path is CPU-testable; bypass only the final SpConv tensor
    # constructor, which is separately exercised by the GPU inference smoke.
    monkeypatch.setattr(Point, "sparsify", lambda self, pad=96: self)
    coords = torch.tensor([[x, y, z] for z in range(2) for y in range(3) for x in range(3)])
    n = len(coords)
    batch = torch.zeros(n, dtype=torch.long)
    centers = coords.float()
    cache = PTv3InferenceTopology.build(coords, centers, batch, [4, 4, 4], ["z", "z-trans"],
                                        3, scene_count=1, level="multilevel")
    top = cache.make_point(torch.randn(n, 8), torch.zeros(n, 4))
    top.offset = torch.tensor([n])
    attn = SerializedAttention(8, 2, 4, enable_flash=False)
    attn.patch_size = 4
    reference_pad = attn.get_padding_and_inverse(top)
    fresh = cache.make_point(torch.randn(n, 8), torch.zeros(n, 4)); fresh.offset = torch.tensor([n])
    reused_pad = attn.get_padding_and_inverse(fresh)
    assert all(torch.equal(a, b) for a, b in zip(reference_pad, reused_pad))
    assert len(cache.attention) == 1

    pool = SerializedPooling(8, 8, stride=2, shuffle_orders=False, traceable=True)
    pool.norm = None
    pool.act = None
    baseline = Point({"feat": torch.randn(n, 8), "coord": centers, "grid_coord": coords.int(),
                      "batch": batch, "offset": torch.tensor([n]), "sparse_shape": [4, 4, 4]})
    baseline.serialization(order=["z", "z-trans"], depth=3, shuffle_orders=False)
    expected = pool(baseline)

    cached_input = cache.make_point(baseline.feat.clone(), torch.zeros(n, 4))
    cached_input.serialization(order=["z", "z-trans"], depth=3, shuffle_orders=False)
    cached_input.offset = torch.tensor([n])
    actual = pool(cached_input)
    assert torch.equal(expected.feat, actual.feat)
    assert torch.equal(expected.coord, actual.coord)
    assert torch.equal(expected.grid_coord, actual.grid_coord)
    assert torch.equal(expected.serialized_order, actual.serialized_order)
    assert torch.equal(expected.pooling_inverse, actual.pooling_inverse)
    assert len(cache.pooling) == 1

    next_input = cache.make_point(baseline.feat.clone(), torch.zeros(n, 4))
    next_input.serialization(order=["z", "z-trans"], depth=3, shuffle_orders=False)
    next_input.offset = torch.tensor([n])
    reused = pool(next_input)
    assert torch.equal(actual.feat, reused.feat)
