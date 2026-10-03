import torch

from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder, keops_knn


def test_precomputed_knn_path_matches_default_encoder_path():
    torch.manual_seed(7)
    points = torch.rand(1, 64, 3) * 40.0 - 20.0
    encoder = DynamicKNNPillarEncoder(out_channels=8, grid_size=16, k=16).eval()
    with torch.no_grad():
        indices = keops_knn(points, points, encoder.k)[1]
        expected = encoder(points)
        actual = encoder.forward_with_knn(points, indices)
        through_public_forward = encoder(points, knn_indices=indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(through_public_forward, expected, rtol=0, atol=0)
