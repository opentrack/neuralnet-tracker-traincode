import torch

from trackertraincode.neuralnets.math import affinevecmul, random_choice
from trackertraincode.neuralnets.affine2d import Affine2d, roi_normalizing_transform


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