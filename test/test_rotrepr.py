import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from typing import Type

from trackertraincode.neuralnets.torchquaternion import positivereal
from trackertraincode.neuralnets.rotrepr import Mat33Repr, QuatRepr


@pytest.mark.parametrize("class_", [QuatRepr, Mat33Repr])
def test_rotation_repr(class_: Type):
    def as_rot_repr(rot) -> QuatRepr | Mat33Repr:
        if class_ is QuatRepr:
            return class_(torch.from_numpy(rot.as_quat()))
        else:
            return class_(torch.from_numpy(rot.as_matrix()))

    N = 10
    NP = 7
    rots1 = Rotation.random(num=N)
    rots2 = Rotation.random(num=N)
    points = np.random.randn(N, NP, 3)
    angles = np.random.randn(N)

    # Multiplication / Chaining
    mult_ref = as_rot_repr(rots1 * rots2)
    mult_res = as_rot_repr(rots1).mult(as_rot_repr(rots2))
    torch.testing.assert_close(mult_ref.value, mult_res.value)

    # Transform points
    points_ref = torch.from_numpy(
        np.stack([rots1.apply(points[:, i, :]) for i in range(NP)], axis=1)
    )
    points_res = as_rot_repr(rots1).rotate_points(torch.from_numpy(points))
    torch.testing.assert_close(points_ref, points_res)

    # Create Rotation
    res_xrot = class_.make_rotate_x(torch.from_numpy(angles))
    ref_xrot = as_rot_repr(
        Rotation.from_rotvec(np.concatenate([angles[:, None], np.zeros((N, 2))], axis=-1))
    )
    torch.testing.assert_close(res_xrot.value, ref_xrot.value)

    # Rotation differences
    # delta_ref = torch.from_numpy((rots1.inv() * rots2).as_rotvec())
    # delta_res = as_rot_repr(rots2).relative_to(as_rot_repr(rots1))
    # torch.testing.assert_close(delta_ref, delta_res)

    # As quat
    torch.testing.assert_close(
        positivereal(torch.from_numpy(rots1.as_quat())), positivereal(as_rot_repr(rots1).as_quat())
    )
