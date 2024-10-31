
import torch
import torch.nn as nn

from trackertraincode.neuralnets.negloglikelihood import (
    FeaturesAsTriangularScale, 
    TangentSpaceRotationDistribution, 
    FeaturesAsDiagonalScale)


def test_tangent_space_rotation_distribution():
    with torch.autograd.set_detect_anomaly(True):
        B = 5
        q = torch.rand((B, 4), requires_grad=True)
        cov_features = torch.rand((B, 6), requires_grad=True)
        r = torch.rand((B, 4))
        cov_converter = FeaturesAsTriangularScale(6,3)
        dist = TangentSpaceRotationDistribution(q, cov_converter(cov_features))
        val = dist.log_prob(r).sum()
        val.backward()
        assert q.grad is not None
        assert cov_features.grad is not None


def test_feature_to_variance_mapping():
    B = 1
    N = 7
    M = 3
    q = torch.zeros((B, N), requires_grad=True)
    m = FeaturesAsDiagonalScale(N,M).eval()
    v = m(q)
    val = v.sum()
    val.backward()
    assert next(iter(m.parameters())).grad is not None
    assert q.grad is not None
    torch.testing.assert_close(v, torch.ones((B,M)), atol=0.1, rtol=0.1)

def test_feature_as_triangular_cov_factor():
    B = 1
    N = 7
    M = 3
    q = torch.zeros((B,N), requires_grad=True)
    m = FeaturesAsTriangularScale(N, M).eval()
    v = m(q)
    val = v.sum()
    val.backward()
    assert next(iter(m.parameters())).grad is not None
    assert q.grad is not None
    torch.testing.assert_close(v, torch.eye(M)[None,...], atol=0.1, rtol=0.1)