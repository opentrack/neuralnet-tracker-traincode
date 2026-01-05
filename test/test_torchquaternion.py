import numpy as np
from numpy.typing import NDArray
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
    quat_average,
)


_rotation_vals = Rotation.concatenate(
    [
        Rotation.random(num=100),
        Rotation.identity(num=1),
        Rotation.from_rotvec([[np.pi, 0.0, 0.0], [0.0, np.pi, 0.0], [0.0, 0.0, np.pi]]),
    ]
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

    angle_difference = geodesicdistance(
        torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat())
    )
    expected_angle_diff = (rots.inv() * rots2).magnitude()
    assert np.allclose(angle_difference, expected_angle_diff)


@pytest.mark.parametrize(
    "rots",
    [
        _rotation_vals,
        _rotation_vals[0],
    ],
)
def test_quaternion_from_matrix(rots: Rotation):
    input_mat = rots.as_matrix()
    expected_quat = positivereal(torch.from_numpy(rots.as_quat()))
    output_quat = from_matrix(torch.from_numpy(input_mat))
    np.testing.assert_allclose(expected_quat.numpy(), output_quat.numpy(), rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize("rots", [_rotation_vals, _rotation_vals[0]])
def test_from_matrix_backprop(rots: Rotation):
    torch.autograd.set_detect_anomaly(True)
    mat = torch.from_numpy(rots.as_matrix()).to(torch.float32)
    mat.requires_grad = True
    q = from_matrix(mat)
    q.sum().backward()

    assert mat.grad is not None
    # print (mat.grad.abs().mean())
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
        Model(),
        (q, p),
        tmp_path
        / "mult_model.onnx",  # model being run  # model input (or a tuple for multiple inputs)
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
        tmp_path / "rotate_model.onnx",
    )


def test_quaternion_from_matrix_onnx_export(tmp_path):
    class Model(torch.nn.Module):
        def __call__(self, m):
            return from_matrix(m)

    m = torch.from_numpy(Rotation.random(10).as_matrix())

    _export_func(
        Model(),  # model being run
        (m,),  # model input (or a tuple for multiple inputs)
        tmp_path / "from_matrix_model.onnx",
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


@pytest.mark.parametrize("use_weights", [True, False])
def test_compute_mean_rotation_with_weights(*, use_weights: bool):
    N = 32
    Noffsets = 1024
    # No convergence for too large scatter. At least not to the expected center. Below 70 deg it work pretty well.
    offset_std = 0.8 * np.pi / 2.0
    max_iter = 16

    rot_means = Rotation.random(N)
    offsets = np.random.normal(size=(N, Noffsets, 3))
    offsets = offsets * offset_std

    rot_batch: list[Rotation] | torch.Tensor = []
    weight_batch: torch.Tensor | list[NDArray[np.floating]] | None = []
    if use_weights:
        for mean in rot_means:
            rots = Rotation.random(num=Noffsets)
            delta = (mean.inv() * rots).as_rotvec()
            x = np.linalg.norm(delta, axis=-1) / offset_std
            weights: NDArray[np.floating] = np.exp(-x * x)
            rot_batch.append(rots)
            weight_batch.append(weights)
    else:
        for mean, offset in zip(rot_means, offsets):
            rot_offset = Rotation.from_rotvec(offset)
            rot = mean * rot_offset
            rot_batch.append(rot)
        weight_batch = None

    quat_batch = torch.from_numpy(np.stack([r.as_quat() for r in rot_batch]))
    weight_batch = torch.from_numpy(np.stack(weight_batch)) if weight_batch else None

    if torch.cuda.is_available():
        quat_batch.to("cuda")
        if isinstance(weight_batch, torch.Tensor):
            weight_batch.to("cuda")

    assert isinstance(weight_batch, (type(None), torch.Tensor))
    mean_quat = quat_average(quat_batch, weight_batch, max_iter=max_iter)

    mean_quat = mean_quat.cpu()
    estimated_means = Rotation.from_quat(mean_quat.numpy())

    deltas = (rot_means.inv() * estimated_means).magnitude()
    print(
        f"Diff from average: {deltas * 180. / np.pi} deg vs exact std of the offsets: {offset_std*180/np.pi}"
    )

    random_pick_deltas = np.linalg.norm(offsets[:, 0, :], axis=-1)
    print(
        f"Average offset {np.average(deltas) * 180. / np.pi} vs random picks {np.average(random_pick_deltas) * 180. / np.pi}"
    )

    assert np.average(deltas) < 0.1 * np.average(random_pick_deltas)
