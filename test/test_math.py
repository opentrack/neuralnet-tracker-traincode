import pytest
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from trackertraincode.neuralnets.math import affinevecmul, random_choice, rotvec_to_matrix
from trackertraincode.neuralnets.affine2d import Affine2d, roi_normalizing_transform


def test_transforms():
    # Multiplying a transform with its inverse should result in identity matrix
    N = 10
    trand = torch.rand((N, 2))
    srand = 0.5 + torch.rand((N,))
    rrand = torch.rand((N,))

    # Test batch computations
    m1 = Affine2d.trs(translations=trand, angles=rrand, scales=srand)
    assert m1.R.shape == (N, 2, 2)
    assert m1.T.shape == (N, 2)
    assert m1.R33.shape == (N, 3, 3)
    m2 = m1.inv()
    out = (m1 @ m2).tensor33()
    assert torch.allclose(out - torch.eye(3), torch.zeros_like(out), atol=1.0e-6)
    assert torch.allclose(m1.scales, srand, atol=1.0e-6)
    assert torch.all(m1.det > 0.0)

    # Test arguments for trs and trs_inv
    for tvec in [None, trand]:
        for svec in [None, srand]:
            for rvec in [None, rrand]:
                if tvec is None and svec is None and rvec is None:
                    continue
                m1 = Affine2d.trs(translations=tvec, angles=rvec, scales=svec)
                m2 = Affine2d.trs_inv(translations=tvec, angles=rvec, scales=svec)
                out = (m1 @ m2).tensor33()
                assert torch.allclose(out - torch.eye(3), torch.zeros_like(out), atol=1.0e-6)

    # Non-batched calculations
    m1 = Affine2d.trs(translations=trand[0], angles=rrand[0], scales=srand[0])
    m2 = m1.inv()
    out = (m1 @ m2).tensor33()
    assert torch.allclose(out - torch.eye(3), torch.zeros_like(out), atol=1.0e-6)
    assert torch.allclose(m1.tensor33()[..., :2, :], m1.m, atol=1.0e-12)

    # Transfer to different device
    assert isinstance(m1.to("cpu").to(torch.float32), Affine2d)

    # Test flip
    m2 = Affine2d.horizontal_flip(torch.tensor(50.0))
    assert torch.allclose(m2.det, torch.tensor(-1.0), atol=1.0e-6)
    out = affinevecmul(m2.tensor(), torch.tensor([100.0, 42.0]))
    assert torch.allclose(out, torch.tensor([0.0, 42]), atol=1.0e-6)

    m1 = Affine2d.range_remap(1, 2, 3, 4)
    assert m1.tensor().dtype == torch.float32
    assert m1.tensor().shape == (2, 3)

    m2 = m1[None, ...]
    assert m2.tensor().shape == (1, 2, 3)

    m1 = Affine2d.trs(translations=trand)
    m3 = m2 @ m1
    assert m3.tensor().shape == m1.tensor().shape


def test_roi_normalizing_transform():
    x0, y0, x1, y1 = 2.0, 3.0, 2.0 + 10.0, 3.0 + 5.0
    roi = torch.tensor([x0, y0, x1, y1])
    tr1 = roi_normalizing_transform(roi)
    assert torch.allclose(
        affinevecmul(tr1.tensor(), torch.tensor([x0, y0])), torch.tensor([-1.0, -1.0])
    )
    assert torch.allclose(
        affinevecmul(tr1.tensor(), torch.tensor([x1, y1])), torch.tensor([1.0, 1.0])
    )
    # Inverse aspect
    x0, y0, x1, y1 = y0, x0, y1, x1
    roi = torch.tensor([x0, y0, x1, y1])
    tr3 = roi_normalizing_transform(roi)
    assert torch.allclose(
        affinevecmul(tr3.tensor(), torch.tensor([x0, y0])), torch.tensor([-1.0, -1.0])
    )
    assert torch.allclose(
        affinevecmul(tr3.tensor(), torch.tensor([x1, y1])), torch.tensor([1.0, 1.0])
    )
    # Now with higher ranks
    x0, y0, x1, y1 = 2.0, 3.0, 2.0 + 10.0, 3.0 + 5.0
    roi = torch.tensor([[[x0, y0, x1, y1], [y0, x0, y1, x1]]])
    trs = roi_normalizing_transform(roi)
    assert torch.allclose(trs.tensor(), torch.stack([tr1.tensor(), tr3.tensor()], dim=0)[None, ...])


def test_random_choice():
    vals = torch.tensor([1.0, 2.0])
    weights = torch.tensor([0.5, 0.5])
    x = random_choice((), vals, weights, True)
    assert x.shape == ()
    x = random_choice((5,), vals, weights, True)
    assert x.shape == (5,)
    x = random_choice((3, 3), vals, weights, True)
    assert x.shape == (3, 3)


@pytest.mark.parametrize("compiled", [False, True])
def test_rotvec_to_matrix(compiled: bool):
    """Test rotvec_to_matrix."""

    np.random.seed(42)

    # Generate batch of random rotation vectors with angles up to 2π
    rotvec_batch_np = np.random.randn(32, 3) * (2 * np.pi)
    rotvec_batch_torch = torch.from_numpy(rotvec_batch_np).float()

    if compiled:
        tested_rotvec_to_matrix = torch.compile(rotvec_to_matrix, fullgraph=True)
    else:
        tested_rotvec_to_matrix = rotvec_to_matrix
    matrices_ours = tested_rotvec_to_matrix(rotvec_batch_torch).numpy()

    matrices_scipy = Rotation.from_rotvec(rotvec_batch_np).as_matrix()

    np.testing.assert_allclose(matrices_ours, matrices_scipy, atol=1e-5)

    # Test zero rotation (identity)
    matrix_zero = tested_rotvec_to_matrix(torch.zeros(1, 3)).numpy()
    np.testing.assert_allclose(matrix_zero, np.eye(3)[None, :, :], atol=1e-6)
