from torch import nn
from torch import Tensor
import torch

import math
from scipy.spatial.transform import Rotation
import pytest
import numpy as np

from trackertraincode.eval import (
    PerspectiveCorrector,
    compute_mean_rotation,
    compute_opal_paper_alignment,
)


def _make_rotations_around_mean(
    mean: Rotation, count: int, spread_deg: float, random_state: np.random.RandomState | None
):
    output = Rotation.identity(num=0)
    while len(output) < count:
        random_rots = Rotation.random(count, random_state)
        mask = (mean.inv() * random_rots).magnitude() < spread_deg * np.pi / 180.0
        random_rots = random_rots[mask]
        output = Rotation.concatenate([output, random_rots])
    return output[:count]


def test_mean_rotation():
    # Setup
    # Warning: It works only for small rotations up to pi/2 at the very most
    center_rot = Rotation.from_euler("XYZ", [20.0, 10.0, 5.0], degrees=True)
    random_rots = _make_rotations_around_mean(
        center_rot, 100, 60.0, np.random.RandomState(seed=123456)
    )
    # Test if the average matches the known mean
    mean = compute_mean_rotation(random_rots)
    assert (mean.inv() * center_rot).magnitude() < 5.0 * np.pi / 180.0


def test_opal_alignment():
    rand_state = np.random.RandomState(seed=123456)
    N = 100
    center1 = Rotation.from_euler("XYZ", [20.0, 10.0, 5.0], degrees=True)
    center2 = Rotation.from_euler("XYZ", [4.0, 2.0, 30.0], degrees=True)
    center3 = Rotation.from_euler("XYZ", [3.0, 5.0, 10.0], degrees=True)
    center4 = Rotation.from_euler("XYZ", [5.0, 20.0, 5.0], degrees=True)
    # rots1, rots2, rots3, rots4 = (Rotation.from_quat(np.broadcast_to(c.as_quat(), (N,4))) for c in [center1,center2,center3,center4])
    rots1, rots2, rots3, rots4 = (
        _make_rotations_around_mean(c, N, 30.0, rand_state)
        for c in [center1, center2, center3, center4]
    )
    rots12 = Rotation.concatenate([rots1, rots2])
    rots34 = Rotation.concatenate([rots3, rots4])
    clusters = np.concatenate([np.full((N,), 0, dtype=np.int32), np.full((N,), 1, dtype=np.int32)])

    print(f"expected delta1: {(center1.inv()*center3).magnitude()*180./np.pi}")
    print(f"expected delta2: {(center2.inv()*center4).magnitude()*180./np.pi}")
    print(f"actual delta: {(rots12[:N].mean().inv() * center3).magnitude()*180./np.pi}")
    print(f"actual delta: {(rots12[N:].mean().inv() * center4).magnitude()*180./np.pi}")

    assert (rots12[:N].mean().inv() * center3).magnitude() > 15.0 * np.pi / 180.0
    assert (rots12[N:].mean().inv() * center4).magnitude() > 30.0 * np.pi / 180.0

    aligned12 = Rotation.from_quat(
        compute_opal_paper_alignment(
            torch.from_numpy(rots12.as_quat()), torch.from_numpy(rots34.as_quat()), clusters
        ).numpy()
    )

    print(f"aligned delta: {(aligned12[:N].mean().inv() * center3).magnitude()*180./np.pi}")
    print(f"aligned delta: {(aligned12[N:].mean().inv() * center4).magnitude()*180./np.pi}")

    assert (aligned12[:N].mean().inv() * center3).magnitude() < 3.0 * np.pi / 180.0
    assert (aligned12[N:].mean().inv() * center4).magnitude() < 3.0 * np.pi / 180.0


def fov_h(fov, aspect):
    # w/h = aspect
    # w/f = 2*tan(fov_w/2)
    # h/f = 2*tan(a)
    # aspect = tan(fov_w/2)/tan(a)
    # -> a = atan(1/aspect * tan(fov_w/2))
    return 2.0 * math.atan(1.0 / aspect * math.tan(fov / 2.0 * math.pi / 180.0)) * 180.0 / math.pi


@pytest.mark.parametrize(
    "fov, image_size, coord, pose, expected",
    [
        # Rotation matches the fov angle when position is at the edge of the screen (horizontal)
        (
            90.0,
            [200, 100],
            [200.0, 50.0, 1.0],
            Rotation.identity(),
            Rotation.from_rotvec([0.0, 45.0, 0.0], degrees=True),
        ),
        # Rotation matches the fov angle when position is at the edge of the screen (vertial)
        (
            90.0,
            [200, 100],
            [100.0, 100.0, 1.0],
            Rotation.identity(),
            Rotation.from_rotvec([-fov_h(90.0, 2.0) / 2.0, 0.0, 0.0], degrees=True),
        ),
        # Returns identity for position in the center
        (90.0, [200, 100], [100.0, 50.0, 1.0], Rotation.identity(), Rotation.identity()),
        # Test if original rotation is considered
        (
            90.0,
            [200, 100],
            [100.0, 50.0, 1.0],
            Rotation.from_rotvec([10.0, 20.0, 30.0], degrees=True),
            Rotation.from_rotvec([10.0, 20.0, 30.0], degrees=True),
        ),
    ],
)
def test_perspective_corrector(fov, image_size, coord, pose, expected):
    corrector = PerspectiveCorrector(fov)
    result = Rotation.from_quat(
        corrector.corrected_rotation(
            image_sizes=torch.as_tensor(image_size, dtype=torch.long),
            coord=torch.as_tensor(coord, dtype=torch.float32),
            pose=torch.from_numpy(pose.as_quat()).to(dtype=torch.float32),
        ).numpy()
    )
    assert Rotation.approx_equal(
        expected, result, atol=0.01, degrees=True
    ), f"Converted to quats: expected = {expected.as_quat()} vs result = {result.as_quat()}"


def test_make_look_at_matrix():
    m = PerspectiveCorrector._make_look_at_matrix(torch.as_tensor([0.0, 0.0, 1.0])).numpy()
    np.testing.assert_allclose(m, np.eye(3))

    SQRT3 = math.sqrt(3.0)
    m = PerspectiveCorrector._make_look_at_matrix(torch.as_tensor([1.0, 1.0, 1.0])).numpy()
    np.testing.assert_allclose(m[:, 2], np.asarray([1.0 / SQRT3, 1.0 / SQRT3, 1.0 / SQRT3]))
    assert np.abs(np.dot(m[:, 0], np.asarray([0.0, 1.0, 0.0]))) < 1.0e-6
    assert m[0, 0] > 0.1
    assert m[1, 1] > 0.1
