import numpy as np
import torch
from scipy.spatial.transform import Rotation
from matplotlib import pyplot
import pytest

from trackertraincode.neuralnets.torchquaternion import (
    from_matrix,
    from_rotvec,
    geodesicdistance,
    iw,
    mult,
    positivereal,
    rotate,
    slerp,
    to_rotvec,
    tomatrix,
    distance,
)


_rotation_vals = Rotation.concatenate(
        [Rotation.random(num=100), Rotation.identity(num=1), Rotation.from_rotvec([[np.pi, 0.0, 0.0],[0.0, np.pi, 0.0],[0.0, 0.0, np.pi]])]
    )


def test_quaternions():
    us = Rotation.from_rotvec(np.random.uniform(0.0, 1.0, size=(7, 3)))
    vs = Rotation.from_rotvec(np.random.uniform(0.0, 1.0, size=(7, 3)))
    q_test = (us * vs).as_quat()
    q_expect = mult(torch.from_numpy(us.as_quat()), torch.from_numpy(vs.as_quat()))
    assert np.allclose(q_test, q_expect)

    us = Rotation.from_rotvec(np.random.uniform(0.0, 1.0, size=(2, 3)))
    vs = np.random.uniform(-1.0, 1.0, size=(4, 1, 3, 1))
    q_test = np.matmul(us.as_matrix()[None, ...], vs)[..., 0]
    q_expect = rotate(torch.from_numpy(us.as_quat()[None, ...]), torch.from_numpy(vs[..., 0]))
    assert np.allclose(q_test, q_expect)

    rots = Rotation.random(10)
    m_test = tomatrix(torch.from_numpy(rots.as_quat())).numpy()
    m_expect = rots.as_matrix()
    assert np.allclose(m_test, m_expect)

    rots = Rotation.random(10)
    q_expect = rots.as_quat()
    q_test = from_rotvec(torch.from_numpy(rots.as_rotvec())).numpy()
    # Flip direction if needed, because q and -q represent the same rotation.
    sign_fix = np.sign(q_expect[:, iw] * q_test[:, iw])
    q_test *= sign_fix[..., None]
    assert np.allclose(q_test, q_expect, rtol=1.0, atol=1.0e-4)

    rotvec_expect = rots.as_rotvec()
    rotvec_test = to_rotvec(torch.from_numpy(rots.as_quat())).numpy()
    assert np.allclose(rotvec_expect, rotvec_test)

    rots2 = Rotation.random(10)

    q_test = slerp(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()), 0.0).numpy()
    assert np.allclose((rots.inv() * Rotation.from_quat(q_test)).magnitude(), 0.0)

    q_test = slerp(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()), 1.0).numpy()
    assert np.allclose((rots2.inv() * Rotation.from_quat(q_test)).magnitude(), 0.0)

    rot_test = Rotation.from_quat(
        slerp(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()), 0.5).numpy()
    )
    assert np.allclose((rots.inv() * rot_test).magnitude(), (rots2.inv() * rot_test).magnitude())
    assert np.allclose((rots.inv() * rot_test).magnitude(), 0.5 * (rots.inv() * rots2).magnitude())

    angle_difference = geodesicdistance(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()))
    expected_angle_diff = (rots.inv() * rots2).magnitude()
    assert np.allclose(angle_difference, expected_angle_diff)


@pytest.mark.parametrize('rots', [
    _rotation_vals,
    _rotation_vals[0],
])
def test_quaternion_from_matrix(rots : Rotation):
    input_mat = rots.as_matrix()
    expected_quat = positivereal(torch.from_numpy(rots.as_quat()))
    output_quat = from_matrix(torch.from_numpy(input_mat))
    np.testing.assert_allclose(expected_quat.numpy(), output_quat.numpy(), rtol=1.e-6, atol=1.e-6)


@pytest.mark.parametrize('rots', [
    _rotation_vals,
    _rotation_vals[0]
])
def test_from_matrix_backprop(rots : Rotation):
    torch.autograd.set_detect_anomaly(True)
    mat = torch.from_numpy(rots.as_matrix()).to(torch.float32)
    mat.requires_grad = True
    q = from_matrix(mat)
    q.sum().backward()

    assert mat.grad is not None
    #print (mat.grad.abs().mean())
    assert torch.isfinite(mat.grad).all()   


def _export_func(model, inputs, filename):
    torch.onnx.export(
        model,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        filename,
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        keep_initializers_as_inputs=False,
        verbose=False,
    )


def test_quaternion_mult_onnx_export(tmp_path):
    class Model(torch.nn.Module):
        def __call__(self, q, p):
            return mult(q, p)

    q = torch.from_numpy(Rotation.random(10).as_quat())
    p = torch.from_numpy(Rotation.random(10).as_quat())

    _export_func(
        Model(), (q, p), tmp_path / 'mult_model.onnx'  # model being run  # model input (or a tuple for multiple inputs)
    )


def test_quaternion_rotate_onnx_export(tmp_path):
    class Model(torch.nn.Module):
        def __call__(self, q, p):
            return rotate(q, p)

    q = torch.from_numpy(Rotation.random(10).as_quat())
    p = torch.from_numpy(np.random.uniform(-10.0, 10.0, size=(10, 3)))

    _export_func(
        Model(),  # model being run
        (q, p),  # model input (or a tuple for multiple inputs)
        tmp_path / 'rotate_model.onnx',
    )


def test_quaternion_from_matrix_onnx_export(tmp_path):
    class Model(torch.nn.Module):
        def __call__(self, m):
            return from_matrix(m)

    m = torch.from_numpy(Rotation.random(10).as_matrix())

    _export_func(
        Model(),  # model being run
        (m,),  # model input (or a tuple for multiple inputs)
        tmp_path / 'from_matrix_model.onnx',
    )


def test_loss():
    rots = Rotation.random(num=50)
    rots1 = Rotation.from_matrix(rots.as_matrix()[np.arange(50).repeat(50)])
    rots2 = Rotation.from_matrix(rots.as_matrix()[np.tile(np.arange(50), 50)])
    delta = (rots1.inv() * rots2).magnitude()

    loss = distance(torch.from_numpy(rots1.as_quat()), torch.from_numpy(rots2.as_quat()))

    np.testing.assert_allclose(loss.numpy(), 0.5 * (1.0 - np.cos(delta)), atol=1.0e-6)

    # pyplot.scatter(delta, loss)
    # pyplot.show() 