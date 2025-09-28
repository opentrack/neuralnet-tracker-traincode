from typing import (
    Callable,
    Set,
    Sequence,
    Union,
    List,
    Tuple,
    Dict,
    Optional,
    NamedTuple,
    Any,
    Literal,
)
import numpy as np
from matplotlib import pyplot
import pytest
import numpy.testing
from functools import partial
from kornia.geometry.subpix import spatial_expectation2d
import torch
import torch.nn.functional as F

import trackertraincode.datatransformation as dtr
from trackertraincode.datatransformation.tensors import (
    croprescale_image_cv2,
    affine_transform_image_cv2,
    DownFilters,
    UpFilters,
    croprescale_image_torch,
    affine_transform_image_torch,
)
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.neuralnets.math import affinevecmul
from trackertraincode.datasets.batch import Batch, Metadata
from trackertraincode.datatransformation.batch.geometric import GeneralFocusRoi

"""
Test consistency between scaling of an image and scaling of landmarks.

We have to think of landmarks as slightly off-center, namely to be located at the
top left of the respective pixel area. That is, assuming the landmarks happen to be
integral values.

Then reducing the image size by factor of x simply amounts to multiplying the landmark
coordinates by x.
"""


@pytest.mark.parametrize(
    "bbox,f,t,bbs, expected",
    [
        ([-10, -10, 10, 10], 1.0, [-1.0, 0.0], 0.3, [-16, -10, 4, 10]),
        ([-10, -10, 10, 10], 1.0, [1.0, 0.0], 0.3, [-4, -10, 16, 10]),
        ([-10, -10, 10, 10], 1.0, [0.0, -1.0], 0.3, [-10, -16, 10, 4]),
        ([-10, -10, 10, 10], 1.0, [0.0, 1.0], 0.3, [-10, -4, 10, 16]),
        ([-10, -10, 10, 10], 2.0, [0.0, 0.0], 0.3, [-20, -20, 20, 20]),
        ([-10, -10, 10, 10], 2.0, [-1.0, 0.0], 0.3, [-36, -20, 4, 20]),
        ([-10, -10, 10, 10], 0.5, [0.0, 0.0], 0.3, [-5, -5, 5, 5]),
        ([-10, -10, 10, 10], 0.5, [-1.0, 0.0], 0.3, [-13, -5, -3, 5]),
    ],
)
def test_compute_view_roi(bbox, f, t, bbs, expected):
    outbox = GeneralFocusRoi._compute_view_roi(
        face_bbox=torch.tensor(bbox, dtype=torch.float32),
        enlargement_factor=torch.tensor(f),
        translation_factor=torch.tensor(t),
        beyond_border_shift=bbs,
    )
    assert outbox.numpy().tolist() == expected


def no_randomization(B, filter_args) -> dtr.batch.RoiFocusRandomizationParameters:
    return dtr.batch.RoiFocusRandomizationParameters(
        scales=torch.tensor(1.0),
        angles=torch.tensor(0.0),
        translations=torch.tensor([0.0, 0.0]),
        **filter_args,
    )


def with_some_similarity_trafo(B, filter_args) -> dtr.batch.RoiFocusRandomizationParameters:
    return dtr.batch.RoiFocusRandomizationParameters(
        scales=torch.tensor(0.75),
        angles=torch.tensor(20.0 * np.pi / 180.0),
        translations=torch.tensor([-0.1, 0.03]),
        **filter_args,
    )


UpDownSampleHint = Literal["up", "down"]


class _TestData(NamedTuple):
    S: int  # size
    R: int  # new size
    batch: Batch  # Contains image, 3d points, and roi.
    tol: float  # Pixel tolerance for point reconstruction based on heatmap.


def make_test_data(scale_up_or_down: UpDownSampleHint):
    """Creates a heatmap with 3 peaks and corresponding 3d points and an roi bounding the full image.

    When downscaling is used then the image is made up of 10x10 blocks, so that the downscaling
    will produce 1 pixel per block so that the points can be reconstructed accurately.

    Otherwise a single pixels will be set per point.
    """
    if scale_up_or_down == "down":
        S = 200
        R = 20
        points = torch.tensor([[15, 15, 0], [45, 35, 0], [85, 85, 0]], dtype=torch.float32)
        points += 50
    else:
        S = 20
        R = 200
        points = torch.tensor([[1, 1, 0], [4, 3, 0], [8, 8, 0]], dtype=torch.float32)
        # For align_corners=False when pixel values are cell centered:
        points += 0.5
        points += 5

    img = torch.zeros((3, 20, 20), dtype=torch.float32)
    # Leave space at border to account for blurring so that the points can be
    # reconstructed from the peaks very precisely.
    img[0, 5 + 1, 5 + 1] = 255.0
    img[1, 5 + 3, 5 + 4] = 255.0
    img[2, 5 + 8, 5 + 8] = 255.0
    if scale_up_or_down == "down":
        img = img.repeat_interleave(10, dim=1).repeat_interleave(10, dim=2)

    # For align_corners=False, the roi subsumes the area from the first
    # to the last pixel completely
    roi = torch.tensor([0.0, 0.0, S, S])

    batch = Batch.from_data_with_categories(
        Metadata(_imagesize=S, batchsize=0),
        {
            "image": (img, dtr.FieldCategory.image),
            "pt3d_68": (points, dtr.FieldCategory.points),
            "roi": (roi, dtr.FieldCategory.roi),
        },
    )
    return _TestData(S, R, batch, 0.01)


def check(td: _TestData, image, points):
    """Check if heatmap and points match."""
    hm = image[None, ...]
    assert hm.shape == (1, len(points), td.R, td.R)
    hm /= hm.sum(dim=[-1, -2], keepdims=True)
    heatmap_points = spatial_expectation2d(hm, normalized_coordinates=False)[0]
    # Because the vertices of the heatmap are centered in the middle of the pixel areas
    heatmap_points += 0.5
    coord_points = points[:, :2]
    for i, (a, b) in enumerate(zip(heatmap_points, coord_points)):
        numpy.testing.assert_allclose(a, b, atol=td.tol, err_msg=f"Mismatch at point {i}")


def vis(td, image, points):
    if 0:
        fig, ax = pyplot.subplots(1, 1)
        extent = [0.0, td.R, td.R, 0.0]
        img = ax.imshow(
            dtr.tensors.ensure_image_nhwc(image).mean(dim=-1),
            interpolation="nearest",
            vmin=0.0,
            extent=extent,
        )
        ax.scatter(*points.T[:2])
        fig.colorbar(img, ax=ax)
        pyplot.show()


up_down_sample_configs = [
    (
        "up",
        {"upfilter": "linear"},
        0.0,
    ),
    (
        "up",
        {"upfilter": "cubic"},
        0.5,
    ),
    (
        "up",
        {"upfilter": "lanczos"},
        0.5,
    ),
    ("down", {"downfilter": "gaussian"}, 0.0),
    ("down", {"downfilter": "hamming"}, 0.0),
    ("down", {"downfilter": "area"}, 0.0),
]


@pytest.mark.parametrize("scaling_way, filter_args, tol", up_down_sample_configs)
def test_scalingtrafo(scaling_way, filter_args, tol):
    td = make_test_data(scaling_way)
    augmentation = dtr.batch.RandomFocusRoi(new_size=td.R)
    augmentation.make_randomization_parameters = partial(no_randomization, filter_args=filter_args)
    td = td._replace(tol=td.tol + tol)
    result = augmentation(td.batch)
    vis(td, result["image"], result["pt3d_68"])
    check(td, result["image"], result["pt3d_68"])


@pytest.mark.parametrize("scaling_way, filter_args, tol", up_down_sample_configs)
def test_scalingtrafo_with_randomizer(scaling_way, filter_args, tol):
    td = make_test_data(scaling_way)
    augmentation = dtr.batch.RandomFocusRoi(new_size=td.R)
    augmentation.make_randomization_parameters = partial(
        with_some_similarity_trafo, filter_args=filter_args
    )
    result = augmentation(td.batch)
    td = td._replace(tol=td.tol + tol + (0.2 if scaling_way == "down" else 0.5))
    vis(td, result["image"], result["pt3d_68"])
    check(td, result["image"], result["pt3d_68"])


@pytest.mark.parametrize("scaling_way", ["up", "down"])
def test_image_affine_transform(scaling_way: str):
    td = make_test_data(scaling_way)
    tr = Affine2d.range_remap_2d([0, 0], [td.S, td.S], [0, 0], [td.R, td.R])
    tr = (
        Affine2d.trs(
            translations=torch.tensor([3.0, -td.R * 0.45]),
            angles=torch.tensor(20.0 * np.pi / 180.0),
            scales=torch.tensor(1.5),
        )
        @ tr
    )
    td = td._replace(tol=1.0)
    cv2result = affine_transform_image_cv2(td.batch["image"], tr, td.R)
    torchresult = affine_transform_image_torch(td.batch["image"], tr, td.R)
    points = affinevecmul(tr.tensor(), td.batch["pt3d_68"][..., :2])
    assert torch.sqrt(torch.nn.functional.mse_loss(cv2result, torchresult)).item() <= (
        2.0 if scaling_way == "up" else 10.0
    )
    vis(td, cv2result, points)
    check(td, cv2result, points)
    check(td, torchresult, points)


@pytest.mark.parametrize("scaling_way", ["up", "down"])
def test_image_crop_rescale(scaling_way: str):
    td = make_test_data(scaling_way)
    td.batch["roi"] = td.batch["roi"].to(torch.int32)
    tr = Affine2d.range_remap_2d(td.batch["roi"][:2], td.batch["roi"][2:], [0, 0], [td.R, td.R])
    cv2result = croprescale_image_cv2(td.batch["image"], roi=td.batch["roi"], new_size=(td.R, td.R))
    torchresult = croprescale_image_torch(
        td.batch["image"], roi=td.batch["roi"], new_size=(td.R, td.R)
    )
    points = affinevecmul(tr.tensor(), td.batch["pt3d_68"][..., :2])
    assert torch.sqrt(torch.nn.functional.mse_loss(cv2result, torchresult)).item() <= 7.0
    vis(td, cv2result, points)
    check(td, cv2result, points)
    check(td, torchresult, points)
