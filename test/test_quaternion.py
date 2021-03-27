from scipy.spatial.transform import Rotation
from neuralnets.torchquaternion import mult, rotate

import torch
import numpy as np
    

if __name__ == '__main__':
        us = Rotation.from_rotvec(np.random.uniform(0.,1.,size=(7,3)))
        vs = Rotation.from_rotvec(np.random.uniform(0.,1.,size=(7,3)))
        q_test = (us * vs).as_quat()
        q_expect = mult(torch.from_numpy(us.as_quat()), torch.from_numpy(vs.as_quat()))
        assert np.allclose(q_test, q_expect)

        us = Rotation.from_rotvec(np.random.uniform(0.,1.,size=(2,3)))
        vs = np.random.uniform(-1., 1., size=(4,1,3,1))
        q_test = np.matmul(us.as_matrix()[None,...], vs)[...,0]
        q_expect = rotate(torch.from_numpy(us.as_quat()[None,...]), torch.from_numpy(vs[...,0]))
        assert np.allclose(q_test, q_expect)