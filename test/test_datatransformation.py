import torch
import pytest

from trackertraincode.datatransformation.sample_geometric import GeneralFocusRoi

@pytest.mark.parametrize('bbox,f,t,bbs, expected',[
    ([-10,-10,10,10], 1., [-1.,0.], 0.3, [-16,-10,4,10]),
    ([-10,-10,10,10], 1., [ 1.,0.], 0.3, [-4,-10,16,10]),
    ([-10,-10,10,10], 1., [0.,-1.], 0.3, [-10,-16,10,4]),
    ([-10,-10,10,10], 1., [0., 1.], 0.3, [-10,-4,10,16]),
    ([-10,-10,10,10], 2., [0., 0.], 0.3, [-20,-20,20,20]),
    ([-10,-10,10,10], 2., [-1., 0.], 0.3, [-36,-20,4,20]),
    ([-10,-10,10,10], 0.5, [0., 0.], 0.3, [-5,-5,5,5]),
    ([-10,-10,10,10], 0.5, [-1., 0.], 0.3, [-13,-5,-3,5]),
])
def test_compute_view_roi(bbox, f, t, bbs, expected):
    outbox = GeneralFocusRoi._compute_view_roi(
        face_bbox = torch.tensor(bbox, dtype=torch.float32),
        enlargement_factor = torch.tensor(f),
        translation_factor = torch.tensor(t),
        beyond_border_shift = bbs)
    assert outbox.numpy().tolist() == expected