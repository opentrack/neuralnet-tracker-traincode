from scipy.spatial.transform import Rotation
from trackertraincode.neuralnets.math import affinevecmul, random_choice
from trackertraincode.neuralnets.torchquaternion import mult, rotate, tomatrix, from_rotvec, iw, to_rotvec, slerp
from trackertraincode.neuralnets.affine2d import Affine2d, roi_normalizing_transform

import torch
import numpy as np
    

def test_quaternions():
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

        rots = Rotation.random(10)
        m_test = tomatrix(torch.from_numpy(rots.as_quat())).numpy()
        m_expect = rots.as_matrix()
        assert np.allclose(m_test, m_expect)

        rots = Rotation.random(10)
        q_expect = rots.as_quat()
        q_test = from_rotvec(torch.from_numpy(rots.as_rotvec())).numpy()
        # Flip direction if needed, because q and -q represent the same rotation.
        sign_fix = np.sign(q_expect[:,iw]*q_test[:,iw])
        q_test *= sign_fix[...,None]
        assert np.allclose(q_test, q_expect, rtol = 1., atol = 1.e-4)

        rotvec_expect = rots.as_rotvec()
        rotvec_test = to_rotvec(torch.from_numpy(rots.as_quat())).numpy()
        assert np.allclose(rotvec_expect, rotvec_test)

        rots2 = Rotation.random(10)
        
        q_test = slerp(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()), 0.).numpy()
        assert np.allclose((rots.inv()*Rotation.from_quat(q_test)).magnitude(),0.)

        q_test = slerp(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()), 1.).numpy()
        assert np.allclose((rots2.inv()*Rotation.from_quat(q_test)).magnitude(),0.)

        rot_test = Rotation.from_quat(slerp(torch.from_numpy(rots.as_quat()), torch.from_numpy(rots2.as_quat()), .5).numpy())
        np.allclose((rots.inv()*rot_test).magnitude() ,(rots2.inv()*rot_test).magnitude())
        np.allclose((rots.inv()*rot_test).magnitude(), 0.5*(rots.inv()*rots2).magnitude())


def test_transforms():
        # Multiplying a transform with its inverse should result in identity matrix
        N = 10
        trand = torch.rand((N,2))
        srand = 0.5 + torch.rand((N,))
        rrand = torch.rand((N,))

        # Test batch computations
        m1 = Affine2d.trs(translations=trand, angles=rrand, scales=srand)
        assert m1.R.shape == (N,2,2)
        assert m1.T.shape == (N,2)
        assert m1.R33.shape ==  (N,3,3)
        m2 = m1.inv()
        out = (m1 @ m2).tensor33()
        assert torch.allclose(out - torch.eye(3), torch.zeros_like(out), atol = 1.e-6)
        assert torch.allclose(m1.scales, srand, atol=1.e-6)
        assert torch.all(m1.det > 0.)

        # Test arguments for trs and trs_inv
        for tvec in [ None, trand ]:
           for svec in [ None, srand ]:
              for rvec in [ None, rrand ]:
                if tvec is None and svec is None and rvec is None:
                        continue
                m1 = Affine2d.trs(translations=tvec, angles=rvec, scales=svec)
                m2 = Affine2d.trs_inv(translations=tvec, angles=rvec, scales=svec)
                out = (m1 @ m2).tensor33()
                assert torch.allclose(out - torch.eye(3), torch.zeros_like(out), atol = 1.e-6)

        # Non-batched calculations
        m1 = Affine2d.trs(translations=trand[0], angles=rrand[0], scales=srand[0])
        m2 = m1.inv()
        out = (m1 @ m2).tensor33()
        assert torch.allclose(out - torch.eye(3), torch.zeros_like(out), atol = 1.e-6)
        assert torch.allclose(m1.tensor33()[...,:2,:], m1.m, atol = 1.e-12)

        # Transfer to different device
        assert isinstance(m1.to('cpu').to(torch.float32), Affine2d)

        # Test flip
        m2 = Affine2d.horizontal_flip(torch.tensor(50.))
        assert torch.allclose(m2.det, torch.tensor(-1.), atol=1.e-6)
        out =  affinevecmul(m2.tensor(), torch.tensor([ 100., 42.]))
        assert torch.allclose(out, torch.tensor([ 0., 42 ]), atol=1.e-6)

        m1 = Affine2d.range_remap(1, 2, 3, 4)
        assert m1.tensor().dtype == torch.float32
        assert m1.tensor().shape == (2,3)

        m2 = m1[None,...]
        assert m2.tensor().shape == (1,2,3)

        m1 = Affine2d.trs(translations=trand)
        m3 = m2 @ m1
        assert m3.tensor().shape == m1.tensor().shape


def test_roi_normalizing_transform():
        x0, y0, x1, y1 = 2.,3.,2.+10.,3.+5.
        roi = torch.tensor([x0,y0,x1,y1])
        tr1 = roi_normalizing_transform(roi)
        assert torch.allclose(affinevecmul(tr1.tensor(), torch.tensor([x0,y0])), torch.tensor([-1.,-1.]))
        assert torch.allclose(affinevecmul(tr1.tensor(), torch.tensor([x1,y1])), torch.tensor([1.,1.]))
        # Inverse aspect
        x0, y0, x1, y1 = y0,x0, y1, x1
        roi = torch.tensor([x0,y0,x1,y1])
        tr3 = roi_normalizing_transform(roi)
        assert torch.allclose(affinevecmul(tr3.tensor(), torch.tensor([x0,y0])), torch.tensor([-1.,-1.]))
        assert torch.allclose(affinevecmul(tr3.tensor(), torch.tensor([x1,y1])), torch.tensor([ 1.,1.]))
        # Now with higher ranks
        x0, y0, x1, y1 = 2.,3.,2.+10.,3.+5.
        roi = torch.tensor([[
              [x0,y0,x1,y1],
              [y0,x0,y1,x1]
              ]])
        trs = roi_normalizing_transform(roi)
        assert torch.allclose(trs.tensor(), torch.stack([tr1.tensor(), tr3.tensor()],dim=0)[None,...])


def test_random_choice():
        vals = torch.tensor([1., 2.])
        weights =  torch.tensor([0.5, 0.5])
        x = random_choice((), vals, weights, True)
        assert x.shape == ()
        x = random_choice((5,), vals, weights, True)
        assert x.shape == (5,)
        x = random_choice((3,3), vals, weights, True)
        assert x.shape == (3,3)


if __name__ == '__main__':
        test_quaternions()
        test_transforms()
        test_random_choice()
        test_roi_normalizing_transform()