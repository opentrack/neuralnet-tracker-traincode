from scipy.spatial.transform import Rotation
import torch
import numpy as np
from matplotlib import pyplot

import trackertraincode.neuralnets.torch6drotation as t6dr


def test_loss():
    rots = Rotation.random(num=50)
    rots1 = Rotation.from_matrix(rots.as_matrix()[np.arange(50).repeat(50)])
    rots2 = Rotation.from_matrix(rots.as_matrix()[np.tile(np.arange(50), 50)])
    delta = (rots1.inv() * rots2).magnitude()

    loss = t6dr.rotation_distance_loss(
        torch.from_numpy(rots1.as_matrix()), torch.from_numpy(rots2.as_matrix())
    )

    np.testing.assert_allclose(loss.numpy(), 0.5 * (1.0 - np.cos(delta)), atol=1.0e-6)

    # pyplot.scatter(delta, loss)
    # pyplot.show()


def test_conversions():
    ref = Rotation.random(num=10).as_matrix()

    z = t6dr.frommatrix(torch.from_numpy(ref))
    assert z.shape[-1] == 6

    m = t6dr._reshape_to_vectors(z)
    assert m.shape[-2:] == (2, 3)
    np.testing.assert_allclose(m.numpy(), ref[:, :2, :])

    m = t6dr.tomatrix(z)
    assert m.shape[-2:] == (3, 3)
    np.testing.assert_allclose(m.numpy(), ref, atol=1.0e-6)

    z = torch.randn((10, 6))
    m = t6dr.tomatrix(z)
    np.testing.assert_allclose(torch.linalg.det(m), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(m @ m.mT, torch.eye(3).expand_as(m), atol=2.0e-5)

    z = torch.as_tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    m = t6dr.tomatrix(z)
    np.testing.assert_allclose(m, torch.eye(3))

    z = torch.as_tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    m = t6dr.tomatrix(z)
    np.testing.assert_allclose(m, torch.eye(3))

    z = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m = t6dr.tomatrix(z)
    np.testing.assert_allclose(m, torch.eye(3))


def test_orthonormality_loss():
    val = t6dr.frommatrix(torch.from_numpy(Rotation.random(num=2).as_matrix()))
    loss = t6dr.orthonormality_loss(val)
    np.testing.assert_allclose(loss, torch.zeros((2,)), atol=1.0e-6)

    val = torch.as_tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    loss2 = t6dr.orthonormality_loss(val)
    assert loss2.item() > 1.0e-2

    val = torch.as_tensor([10.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    loss3 = t6dr.orthonormality_loss(val)
    assert loss3.item() > loss2.item() + 1.0e-2
