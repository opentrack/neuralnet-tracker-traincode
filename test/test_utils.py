import numpy as np
from scipy.spatial.transform import Rotation

from trackertraincode import utils


def test_affine3d():
    R1 = Rotation.random()
    t1 = np.random.rand(3)
    inv = utils.affine3d_inv((R1, t1))
    R2, t2 = utils.affine3d_chain((R1, t1), inv)
    np.testing.assert_allclose(R2.as_matrix(), np.eye(3), atol=1.0e-15)
    np.testing.assert_allclose(t2, 0.0, atol=1.0e-9)


def test_as_hpb():
    R = Rotation.random(num=100)
    hpb = utils.as_hpb(R)
    R_back = utils.from_hpb(hpb)
    np.testing.assert_allclose(R.as_matrix(), R_back.as_matrix())


def test_aflw_rotation_conversion():
    rs = Rotation.random(num=100)
    yprs = utils.inv_aflw_rotation_conversion(rs)
    rs_back = utils.aflw_rotation_conversion(*yprs.T)
    np.testing.assert_allclose(rs_back.as_matrix(), rs.as_matrix())
