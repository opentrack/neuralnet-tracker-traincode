import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
import random
import torch
from torch import Tensor
import torch.nn.functional as F

from trackertraincode.pipelines import Batch
from trackertraincode.datasets.batch import Metadata
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories

from trackertraincode.datatransformation.tensors.image_geometric_cv2 import (
    affine_transform_image_cv2,
    croprescale_image_cv2,
    DownFilters,
    UpFilters,
)
from trackertraincode.datatransformation.tensors.affinetrafo import (
    apply_affine2d,
    position_normalization,
    position_unnormalization,
)


class RoiFocusRandomizationParameters(NamedTuple):
    scales: torch.Tensor  # Shape B
    angles: torch.Tensor  # Shape B
    translations: torch.Tensor  # Shape (B, 2)
    intermediate_resolution: int
    upfilter: Optional[UpFilters] = None
    downfilter: Optional[DownFilters] = None


def _compute_view_roi(
    face_bbox: torch.Tensor,
    enlargement_factor: torch.Tensor,
    translation_factor: torch.Tensor,
    beyond_border_shift: float,
    output_aspect: float | int = 1,
):
    """
    Computes the expanded and shifted ROI based on the face bounding box.

    Case 1: small roi
            |--- bbox ----|
            |-roi-|
            <-> At most [beyond_border_shift] of ROI sidelength
    Case 2: large roi
            |--- bbox ----|
                    |-------- roi -------|
                <--> At most [beyond_border_shift] of bounding box side lenght

    Args:
        enlargement_factor: By how much the face bounding box is scaled up
        translation_factor: Random number between -1 and 1 indicating the movement of the face roi within the expanded roi
        beyond_border_shift: Controls the length up to which which ROI and original BBOX may not intersect.
    """
    assert (
        face_bbox.shape[:-1] == enlargement_factor.shape
    ), f"Bad bounding box shape {face_bbox.shape}"
    assert face_bbox.shape[:-1] == translation_factor.shape[:-1]
    x0, y0, x1, y1 = face_bbox.unbind(-1)
    rx, ry = translation_factor.unbind(-1)
    # Size and center of the BBox.
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    cx = 0.5 * (x1 + x0)
    cy = 0.5 * (y1 + y0)
    # Size of the expanded ROI.
    if output_aspect == 1:
        new_w = new_h = torch.maximum(bbox_w, bbox_h) * enlargement_factor
    else:
        width_based_sizes = (
            torch.stack([bbox_w, bbox_w / output_aspect], dim=0) * enlargement_factor
        )
        height_based_sizes = (
            torch.stack([bbox_h * output_aspect, bbox_h], dim=0) * enlargement_factor
        )
        new_w, new_h = torch.where(
            (bbox_w > output_aspect * bbox_h).expand_as(width_based_sizes),
            width_based_sizes,
            height_based_sizes,
        )
    wiggle_room_x = 0.5 * torch.abs(new_w - bbox_w) + beyond_border_shift * torch.minimum(
        new_w, bbox_w
    )
    wiggle_room_y = 0.5 * torch.abs(new_h - bbox_h) + beyond_border_shift * torch.minimum(
        new_h, bbox_h
    )
    tx = wiggle_room_x * rx
    ty = wiggle_room_y * ry
    x0 = cx - new_w * 0.5 + tx
    x1 = cx + new_w * 0.5 + tx
    y0 = cy - new_h * 0.5 + ty
    y1 = cy + new_h * 0.5 + ty
    new_roi = torch.stack([x0, y0, x1, y1], dim=-1)
    return new_roi


class CropAugmentationHelpers:
    def _make_crop_transform(
        self,
        B: tuple[int, ...],
        new_roi: torch.Tensor,
        new_size: int | tuple[int, int] | float | tuple[float, float],
    ):
        """The transform maps points from the original image to local coordinates in the crop.
        Uses absolute coordinates, i.e. not normalized.

        Args:
            new_roi: The crop rectangle in the original image
            new_size: Resampled resolution of the crop.
        """
        max_output_coord = torch.atleast_1d(
            torch.as_tensor(new_size, device=new_roi.device, dtype=torch.float32)
        ).expand(*B, 2)
        return Affine2d.range_remap_2d(
            inmin=new_roi[..., :2].to(torch.float32),
            inmax=new_roi[..., 2:].to(torch.float32),
            outmin=torch.zeros(B + (2,), device=new_roi.device, dtype=torch.float32),
            outmax=max_output_coord,
        )

    def _center_rotation_tr(self, rotations: Tensor, image_size: int | tuple[int, int]):
        # TODO: Complete implementation for batches
        w, h = (image_size, image_size) if isinstance(image_size, int) else image_size
        tr_norm = position_normalization(w, h).to(device=rotations.device)
        tr_rot = Affine2d.trs(angles=rotations)
        tr_denorm = position_unnormalization(w, h).to(device=rotations.device)
        return tr_denorm @ tr_rot @ tr_norm


class FocusRoi(CropAugmentationHelpers):
    def __init__(
        self,
        new_size: int | tuple[int, int],
        extent_factor,
        roi_variable: str | None = "roi",
        insert_backtransform=False,
    ):
        self._new_size = new_size
        self._extent_factor = extent_factor
        self._roi_variable = roi_variable
        self._insert_backtransform = insert_backtransform
        self._new_aspect = 1 if isinstance(new_size, int) else new_size[0] / new_size[1]
        if roi_variable is None:

            def roi_accessor(sample: Batch):  # type: ignore[reportRedeclaration]
                B = sample.meta.prefixshape
                w, h = sample.meta.image_wh
                roi = torch.as_tensor((0, 0, w, h), device=sample.device)
                roi = roi.expand(*B, 4)
                return roi

        else:

            def roi_accessor(sample: Batch):
                return sample[roi_variable]

        self._roi_accessor = roi_accessor

    def _apply_rescale_and_crop(
        self,
        sample: Batch,
        view_roi: torch.Tensor,
        new_size: int | tuple[int, int],
        insert_backtransform: bool,
    ):
        W, H = sample.meta.image_wh
        B = sample.meta.prefixshape

        tr = self._make_crop_transform(B, view_roi, new_size)

        image_transform_function = lambda img: croprescale_image_cv2(
            img, view_roi, new_size, downfilter="area", upfilter="cubic"
        )
        for k, v in sample.items():
            c = sample.get_category(k)
            if c == FieldCategory.image:
                sample[k] = image_transform_function(v)
            else:
                sample[k] = apply_affine2d(tr, k, v, c)

        if insert_backtransform:
            sample["image_backtransform"] = tr.inv().tensor()
            sample["image_original_size"] = torch.tensor((W, H), dtype=torch.int32)

        sample.meta._imagesize = new_size
        return sample

    def __call__(self, sample: Batch):
        B = sample.meta.prefixshape
        roi = self._roi_accessor(sample)

        scales = torch.full(B, self._extent_factor)
        translations = torch.zeros(B + (2,))

        crop_roi = _compute_view_roi(
            roi, scales, translations, beyond_border_shift=0.0, output_aspect=self._new_aspect
        )
        return self._apply_rescale_and_crop(
            sample, crop_roi, self._new_size, self._insert_backtransform
        )


class RandomFocusRoi(CropAugmentationHelpers):
    def __init__(
        self,
        new_size: int | tuple[int, int],
        rotation_aug_angle: float,
        extension_factor: float,
        roi_variable: str | None = "roi",
    ):
        w, h = (new_size, new_size) if isinstance(new_size, int) else new_size
        self.new_size_wh = (w, h)
        self._new_aspect = 1 if isinstance(new_size, int) else new_size[0] / new_size[1]
        self.max_beyond_border_shift = 0.3
        self.rotation_aug_angle = rotation_aug_angle
        self.extension_factor = extension_factor
        if roi_variable is None:

            def roi_accessor(sample: Batch):  # type: ignore[reportRedeclaration]
                B = sample.meta.prefixshape
                w, h = sample.meta.image_wh
                roi = torch.as_tensor((0, 0, w, h), device=sample.device)
                roi = roi.expand(*B, 4)
                return roi

        else:

            def roi_accessor(sample: Batch):
                return sample[roi_variable]

        self._roi_accessor = roi_accessor

    def _sample_roi_randomization(self, B: tuple) -> RoiFocusRandomizationParameters:
        scales = torch.randn(size=B).mul(0.1).clip(-0.5, 0.5).add(self.extension_factor)
        translations = torch.randn(size=B + (2,)).mul(0.5).clip(-1.0, 1.0)
        angles = self._pick_angles(B, self.rotation_aug_angle)
        intermediate_width = int(np.rint(np.random.uniform(0.5, 4.0) * self.new_size_wh[0]))
        return RoiFocusRandomizationParameters(
            scales=scales,
            angles=angles,
            translations=translations,
            intermediate_resolution=intermediate_width,
            upfilter=random.choice(["linear", "cubic", "lanczos"]),
            downfilter=random.choice(["area", "gaussian", "hamming"]),
        )

    def _pick_angles(self, B: tuple, angle: float):
        angles = torch.rand(B) * np.pi * angle / 180.0
        angles *= torch.from_numpy(np.random.choice([-1.0, 1.0], size=B, replace=False))
        angles *= torch.from_numpy(
            np.random.choice([0.0, 1.0], size=B, replace=False, p=[2.0 / 3, 1.0 / 3])
        )
        return angles

    def __call__(self, sample: Batch):
        B = sample.meta.prefixshape
        roi = self._roi_accessor(sample)

        params: RoiFocusRandomizationParameters = self._sample_roi_randomization(B)

        view_roi = _compute_view_roi(
            roi, params.scales, params.translations, self.max_beyond_border_shift, self._new_aspect
        )

        intermediate_w = params.intermediate_resolution
        intermediate_h = params.intermediate_resolution / self._new_aspect
        intermediate_wh = (intermediate_w, intermediate_h)

        view_roi = torch.round(view_roi).to(torch.int32)
        tr = self._make_crop_transform(B, view_roi, intermediate_wh)

        # TODO: this won't work for videos ...
        if params.angles.item() != 0.0:
            # TODO: maybe rotate around face center
            tr = self._center_rotation_tr(params.angles, self.new_size_wh) @ tr
            image_transform_function = lambda img: affine_transform_image_cv2(
                img, tr, params.intermediate_resolution, downfilter="hamming", upfilter="lanczos"
            )
        else:
            image_transform_function = lambda img: croprescale_image_cv2(
                img,
                view_roi,
                params.intermediate_resolution,
                downfilter="hamming",
                upfilter="lanczos",
            )

        image_transform_function2 = lambda img: croprescale_image_cv2(
            img,
            roi=None,
            new_size=self.new_size_wh,
            downfilter=params.downfilter,
            upfilter=params.upfilter,
        )
        tr2 = Affine2d.range_remap_2d(
            inmin=torch.zeros((*B, 2)),
            inmax=torch.as_tensor(intermediate_wh).expand(*B, 2),
            outmin=torch.zeros((*B, 2)),
            outmax=torch.as_tensor(self.new_size_wh).expand(*B, 2),
        )
        label_trafo = tr2 @ tr

        for k, v in sample.items():
            c = sample.get_category(k)
            if c == FieldCategory.image:
                v = image_transform_function(v)
                v = image_transform_function2(v)
                sample[k] = v
            else:
                sample[k] = apply_affine2d(label_trafo, k, v, c)

        sample.meta._imagesize = self.new_size_wh
        return sample


def horizontal_flip_and_rot_90(p_rot: float, sample: Batch):
    assert sample.meta.batchsize == 0
    do_flip = np.random.randint(0, 2) == 0
    rot_dir = np.random.choice([-1, 0, 1], p=[p_rot / 2.0, (1.0 - p_rot), p_rot / 2.0])
    if not do_flip and rot_dir == 0:
        return sample
    sample = copy(sample)
    w, h = sample.meta.image_wh
    tr = Affine2d.identity()
    if rot_dir != 0:
        tr = (
            tr
            @ Affine2d.range_remap_2d([-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [w, h])
            @ Affine2d.trs(angles=torch.tensor(rot_dir * np.pi * 0.5, dtype=torch.float32))
            @ Affine2d.range_remap_2d([0.0, 0.0], [w, h], [-1.0, -1.0], [1.0, 1.0])
        )
    if do_flip:
        tr = tr @ Affine2d.range_remap_2d([0.0, 0.0], [w, h], [w, 0], [0, h])

    for k, v in sample.items():
        c = sample.get_category(k)
        if c in imagelike_categories:
            if do_flip:
                v = torch.flip(v, (-1,))
            if rot_dir != 0:
                v = v.swapaxes(-1, -2)
            if rot_dir == 1:
                v = torch.flip(v, (-1,))
            elif rot_dir == -1:
                v = torch.flip(v, (-2,))
            sample[k] = v
        else:
            sample[k] = apply_affine2d(tr, k, v, c)
    return sample
