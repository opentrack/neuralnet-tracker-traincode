import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple, Dict, Any, Union, List, NamedTuple, Mapping, Literal
from copy import copy
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import tqdm

from torchmetrics import Metric
import torch
from torch import nn
from torch import Tensor

from trackertraincode.datasets.batch import Batch, Metadata
from trackertraincode.neuralnets.affine2d import Affine2d
import trackertraincode.utils as utils
import trackertraincode.datatransformation as dtr
import trackertraincode.neuralnets.torchquaternion as torchquaternion


class InferenceNetwork(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, images: Tensor) -> Dict[str, Tensor]:
        pass

    @property
    @abstractmethod
    def device_for_input(self) -> str:
        pass

    @property
    @abstractmethod
    def input_resolution(self) -> int:
        pass


def _onnx_single_frame_inference(session, batch):
    def _infer_one(img):
        return session.run(None, {'x': img[None, ...]})

    outputs = zip(*(_infer_one(img) for img in batch.numpy()))
    outputs = [np.vstack(o) for o in outputs]
    return outputs


def _onnx_batch_inference(session, batch):
    return session.run(None, {'x': batch.numpy()})


def load_pose_network(filename, device) -> InferenceNetwork:
    if filename.endswith('.onnx'):
        import onnxruntime

        # For showing device placement
        # onnxruntime.set_default_logger_severity(1)
        # onnxruntime.set_default_logger_verbosity(0)

        class OnnxPoseNetwork(InferenceNetwork):
            def __init__(self, modelfile, device):
                providers = {
                    'cpu': [
                        'CPUExecutionProvider',
                    ],
                    # This may look odd but ONNX can decide to run certain ops on the CPU!
                    'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                }[device]
                self.session = onnxruntime.InferenceSession(modelfile, providers=providers)
                # TODO: decide on a names. Also for the OpenTrack plugin.
                namemap = {
                    'pos_size': 'coord',
                    'quat': 'pose',
                    'box': 'roi',
                    'eyes': 'eyeparam',
                    'pos_size_scales': 'coord_scales',
                    'pos_size_std': 'coord_scales',
                    'rotaxis_scales_tril': 'pose_scales_tril',
                    'rotaxis_std': 'pose_scales_tril',
                    'rot_conc_tril': 'pose_conc_tril',
                    'box_scales': 'roi_scales',
                    'box_std': 'roi_scales',
                }
                self.output_names = [namemap.get(o.name, o.name) for o in self.session.get_outputs()]
                # print ("Model version string: ", self.session.get_modelmeta().version, ' has outputs', self.output_names)

                if isinstance(self.session.get_inputs()[0].shape[0], int):
                    self._inference = _onnx_single_frame_inference
                else:
                    self._inference = _onnx_batch_inference

            @property
            def device_for_input(self):
                '''In spite of potentially executing on the gpu, ONNX does not know what a pytorch Tensor on the GPU is.'''
                return 'cpu'

            @property
            def input_resolution(self) -> int:
                return 129  # TODO: this should somehow come from the model file

            def __call__(self, batch):
                outputs = self._inference(self.session, batch)
                outputs = dict(zip(self.output_names, outputs))
                if self.session.get_modelmeta().version not in (2, 3, 4):
                    # Old model needs coordinate conversion
                    quats = outputs['pose']
                    x, y, z = quats[..., 0].copy(), quats[..., 1].copy(), quats[..., 2].copy()
                    quats[..., 0] = -z
                    quats[..., 1] = -y
                    quats[..., 2] = -x
                    outputs['pose'] = quats
                outputs = {k: torch.from_numpy(v) for k, v in outputs.items()}
                return outputs

        return OnnxPoseNetwork(filename, device)
    else:
        import trackertraincode.neuralnets.models

        class PytorchPoseNetwork(InferenceNetwork):
            def __init__(self, modelfile, device):
                net = trackertraincode.neuralnets.models.load_model(modelfile)
                net.eval()
                net.to(device)
                self._net = net
                self._device = device

            @property
            def device_for_input(self):
                return self._device

            @property
            def input_resolution(self) -> int:
                return self._net.input_resolution

            def __call__(self, images):
                return self._net(images)

        return PytorchPoseNetwork(filename, device)


def _apply_backtrafo(backtrafo: Affine2d, batch: Batch):
    batch = copy(batch)
    for k, v in batch.items():
        if batch.get_category(k) in dtr.imagelike_categories:
            continue
        batch[k] = dtr.tensors.apply_affine2d(backtrafo, k, v, batch.get_category(k))
    return batch


class Predictor:
    def __init__(self, net: InferenceNetwork | str, focus_roi_expansion_factor: float = 1.1, device: str | None = None):
        if isinstance(net, InferenceNetwork):
            assert device is None
            self._net = net
        else:
            self._net = load_pose_network(net, device)
        self._roi_focus = dtr.batch.FocusRoi(
            self._net.input_resolution, focus_roi_expansion_factor, insert_backtransform=True
        )

    def _create_sample(self, image: Tensor, roi: Tensor):
        H, W, C = image.shape
        image = dtr.tensors.ensure_image_nchw(image)
        sample = Batch.from_data_with_categories(
            Metadata((H, W), 0),
            {
                'image': (image, dtr.FieldCategory.image),
                'roi': (roi, dtr.FieldCategory.roi),
            },
        )
        return self._roi_focus(sample)

    @property
    def normalize_crop_transform(self):
        return [self._roi_focus, dtr.batch.normalize_batch]

    @torch.no_grad()
    def predict_batch(self, images: List[Tensor], rois: Tensor):
        B = len(images)
        assert rois.shape == (B, 4), f"Bad roi shape: {rois.shape}"
        device = images[-1].device
        batch = [self._create_sample(i, r) for i, r in zip(images, rois)]
        batch = Batch.collate(batch)
        batch = dtr.batch.normalize_batch(batch)

        preds = self._net(dtr.tensors.whiten_image(batch['image']).to(self._net.device_for_input))
        preds = Batch(batch.meta, **preds)
        preds.meta.categories.update(
            {
                'coord': dtr.FieldCategory.xys,
                'pose': dtr.FieldCategory.quat,
                'pt3d_68': dtr.FieldCategory.points,
            }
        )
        preds['image_backtransform'] = batch['image_backtransform'].to(self._net.device_for_input)
        preds = dtr.batch.unnormalize_batch(preds)
        preds = _apply_backtrafo(Affine2d(preds.pop('image_backtransform')), preds)
        preds = preds.to(device)
        return preds

    def evaluate(self, metric: Metric, loader: dtr.SampleBySampleLoader[Batch]):
        bar = tqdm.tqdm(total=len(loader))
        for samples in utils.iter_batched(loader, 128):
            images: list[Tensor] = [s.pop('image') for s in samples]  # Can't collate differently sized images ...
            batch = Batch.collate(samples)
            preds = self.predict_batch(images, batch['roi'])
            # Normally, list of tensors is not allowed but ragged image tensor is required for variable image sizes
            batch['image'] = images
            metric.update(preds, batch)
            bar.update(len(images))
        return metric.compute()

    @torch.no_grad()
    def predict_cropped_normalized_batch(self, images: Tensor) -> Batch:
        preds = self._net(dtr.tensors.whiten_image(images.to(self._net.device_for_input)))
        preds = Batch(
            Metadata(
                images.shape[-2:],
                images.shape[0],
                categories={
                    'coord': dtr.FieldCategory.xys,
                    'pose': dtr.FieldCategory.quat,
                    'pt3d_68': dtr.FieldCategory.points,
                },
            ),
            preds,
        )
        return preds.to(images.device)

    def evaluate_cropped_normalized(self, metric: Metric, loader: dtr.DataLoader):
        bar = tqdm.tqdm(total=len(loader.dataset))
        for batch in loader:
            batch: Batch
            preds = self.predict_cropped_normalized_batch(batch['image'])
            metric.update(preds, batch)
            bar.update(batch.meta.batchsize)
        return metric.compute()


##########################################
## Metrics for evaluation
##########################################

# For reference, here is how the euler angle error metric is computed in 6DRepNet.
# From what I can tell it's the same approach in principle. Ultimately the euler angles
# are compared with the original GT angles from AFLW 2k 3D.
# https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/utils.py#L192
# https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/datasets.py#L58
# https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/test.py#L137
# I checked that their utils.compute_euler_angles_from_rotation_matrices is the inverse
# of their utils.get_R.


class LocalizerBoxMeanSquareErrors(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, pred, sample):
        target = sample['roi']
        mask = sample['hasface'] > self.threshold
        mask &= pred['hasface'] > self.threshold
        err = F.mse_loss(pred['roi'], target[:, :], reduction='none')
        err[~mask, :] = np.nan
        err0 = torch.sum(err[:, :2], dim=1)
        err1 = torch.sum(err[:, 2:], dim=1)
        return torch.cat([err0[:, None], err1[:, None]], dim=1)


class LocalizerIsFaceMatches(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, pred, sample):
        target = sample['hasface']
        score = pred['hasface']
        match = torch.eq(target > self.threshold, score > self.threshold)
        return match


class _SimpleConcatenatingErrorMetric(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("error", default=[], dist_reduce_fx="cat")

    def update(self, preds: Batch, targets: Batch) -> None:
        self.error.append(self.compute_on_batch(preds, targets))

    def compute_on_batch(preds: Batch, targets: Batch) -> Tensor:
        raise NotImplementedError()

    def compute(self) -> Tensor:
        return torch.cat(self.error)


class LabelExtractor(_SimpleConcatenatingErrorMetric):
    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        self._key = key

    def compute_on_batch(self, preds: Batch, targets: Batch) -> Tensor:
        return targets[self._key]


class PredExtractor(_SimpleConcatenatingErrorMetric):
    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        self._key = key

    def compute_on_batch(self, preds: Batch, targets: Batch) -> Tensor:
        return preds[self._key]


class GeodesicError(_SimpleConcatenatingErrorMetric):
    def compute_on_batch(self, preds: Batch, targets: Batch):
        return torchquaternion.geodesicdistance(targets['pose'], preds['pose'])


def _angle_errors(euler1: NDArray, euler2: NDArray):
    v1 = np.concatenate([np.cos(euler1)[..., None], np.sin(euler1)[..., None]], axis=-1)
    v2 = np.concatenate([np.cos(euler2)[..., None], np.sin(euler2)[..., None]], axis=-1)
    angles = np.arccos(np.sum(v1 * v2, axis=-1))
    return angles


def _quat_to_aflw3d_rotations(quats: Tensor) -> NDArray:
    # TODO: Vectorize
    return np.array([utils.inv_aflw_rotation_conversion(q) for q in utils.convert_to_rot(quats.cpu().numpy())])


def _aflw3d_euler_errors(quats1: Tensor, quats2: Tensor) -> Tensor:
    return torch.from_numpy(_angle_errors(_quat_to_aflw3d_rotations(quats1), _quat_to_aflw3d_rotations(quats2))).to(
        device=quats1.device
    )


class EulerAngleErrors(_SimpleConcatenatingErrorMetric):
    def compute_on_batch(self, preds: Batch, targets: Batch):
        return _aflw3d_euler_errors(preds['pose'], targets['pose'])


class NormalizedXYSError(_SimpleConcatenatingErrorMetric):
    def compute_on_batch(self, preds: Batch, targets: Batch):
        coord_target = targets['coord']
        coord = preds['coord']
        x0, y0, x1, y1 = targets['roi'].unbind(-1)
        return torch.abs(coord - coord_target) / (x1 - x0)[:, None]


# Adapted from https://github.com/MCG-NJU/SADRNet/blob/main/src/model/loss.py#L224
def _eval_keypoints(pred: Tensor, gt: Tensor, dims=3):
    # Pred and GT are bached point tensors with shapes
    # B x N x 3, where N is the number of keypoints
    B, N, D = pred.shape
    assert D == 3
    assert pred.shape == gt.shape
    pred = pred.clone()
    gt = gt.clone()
    pred[:, :, 2] -= torch.mean(pred[:, :, 2], dim=-1, keepdim=True)
    gt[:, :, 2] -= torch.mean(gt[:, :, 2], dim=-1, keepdim=True)
    dist = torch.mean(torch.norm(pred[:, :, :dims] - gt[:, :, :dims], dim=-1), dim=-1)
    left = torch.amin(gt[:, :, 0], dim=1)
    right = torch.amax(gt[:, :, 0], dim=1)
    top = torch.amin(gt[:, :, 1], dim=1)
    bottom = torch.amax(gt[:, :, 1], dim=1)
    bbox_size = torch.sqrt((right - left) * (bottom - top))
    dist = dist / bbox_size
    return dist


class UnweightedKptNME(_SimpleConcatenatingErrorMetric):
    def __init__(self, dimensions=3):
        super().__init__()
        self.dims = dimensions

    def compute_on_batch(self, preds: Batch, targets: Batch):
        return _eval_keypoints(preds['pt3d_68'], targets['pt3d_68'], self.dims)


class KptNmeResults(NamedTuple):
    bin_30_nme: float
    bin_60_nme: float
    bin_90_nme: float
    avg_nme: float


class KptNME(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, dimensions=3):
        super().__init__()
        self.add_state("error", default=[], dist_reduce_fx="cat")
        self.add_state("masks", default=[], dist_reduce_fx="cat")
        self.dims = dimensions

    def update(self, preds, targets):
        self.masks.append(self._compute_bin_masks(targets['pose']))
        self.error.append(_eval_keypoints(preds['pt3d_68'], targets['pt3d_68'], self.dims))

    def compute(self) -> Tensor:
        errors = torch.cat(self.error)
        masks = torch.cat(self.masks).unbind(-1)
        nme_by_bins = [torch.mean(errors[m]).item() for m in masks]
        return KptNmeResults(*nme_by_bins, np.average(nme_by_bins).tolist())

    def _compute_bin_masks(self, pose_gt: Tensor) -> Tensor:
        '''Masks for the yaw bins from the literature: 0-30, 30-60, 60-90 deg.

        Return shape (#points, #bins)
        '''
        rot = utils.convert_to_rot(pose_gt.cpu().numpy())
        pyr_gt = np.array([utils.inv_aflw_rotation_conversion(r) for r in rot])
        abs_yaw_deg = np.abs(pyr_gt[:, 1]) * 180.0 / np.pi
        bounds_list = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]
        masks = torch.from_numpy(
            np.stack([((a <= abs_yaw_deg) & (abs_yaw_deg < b)) for (a, b) in bounds_list], axis=-1)
        )
        return masks.to(device=pose_gt.device)


def _compute_displacement(mean_rot: Rotation, rots: Rotation):
    return (mean_rot.inv() * rots).as_rotvec()


def _compute_mean_rotation(rots: Rotation, tol=0.0001, max_iter=100000):
    # Adapted from https://github.com/pcr-upm/opal23_headpose/blob/main/test/evaluator.py#L111C1-L126C27
    # Exclude samples outside the sphere of radius pi/2 for convergence
    rots = rots[rots.magnitude() < np.pi / 2]
    mean_rot = rots[0]
    for _ in range(max_iter):
        displacement = _compute_displacement(mean_rot, rots)
        displacement = np.mean(displacement, axis=0)
        d_norm = np.linalg.norm(displacement)
        if d_norm < tol:
            break
        mean_rot = mean_rot * Rotation.from_rotvec(displacement)
    return mean_rot


def compute_opal_paper_alignment(pose_pred: Tensor, pose_target: Tensor, cluster_ids: NDArray[np.int32]) -> Tensor:
    """Returns updated rotations."""
    assert pose_pred.get_device() == -1  # CPU
    assert pose_target.get_device() == -1  # CPU
    clusters = np.unique(cluster_ids)
    out = torch.empty_like(pose_pred)
    for id_ in clusters:
        mask = cluster_ids == id_
        pred_rot = Rotation.from_quat(pose_pred[mask].numpy())
        target_rot = Rotation.from_quat(pose_target[mask].numpy())
        align_rot = _compute_mean_rotation(target_rot.inv() * pred_rot)
        # print (f"id = {id_}, align = {align_rot.magnitude()*180./np.pi}, {np.count_nonzero(mask)} items")
        # (P (T^-1 * P)^-1 )^-1 T
        #    ----+----
        #        align_rot
        # => (P P^-1 T)^-1 T = Identity
        pred_rot = pred_rot * align_rot.inv()
        out[mask] = torch.from_numpy(pred_rot.as_quat()).to(pose_pred.dtype)
    return out


class PerspectiveCorrector:
    def __init__(self, fov):
        self._fov = fov
        self.f = 1.0 / math.tan(fov * math.pi / 180.0 * 0.5)

    def corrected_rotation(self, image_sizes: Tensor, coord: Tensor, pose: Tensor):
        r'''
            Explanation though top view
                                       ^ face-local z-axis
                         z-axis ^      |   ^ direction under which the CNN "sees" the face through it's crop
                                |     _|__/
                                |    /    \
                                |   | face |
                                |    \ __ /
                                |     /         Note: <----> marks the face crop
                                |    / 
           -----------------------<-x->-------------- screen
                                |  / xy_normalized
                              f | /
                                |/
                        camera  x ------> x-axis

            Thus, it is apparent that the CNN sees the face approximately under an angle spanned by the forward
            direction and the 3d position of the face. The more wide-angle the lense is the stronger the effect. 
            As usual perspective distortion within the crop is neglected.
            Hence, we assume that the detected rotation is given w.r.t to a coordinate system whose z-axis is
            aligned with the position vector as illustrated. Consequently, the resulting pose is simply the
            cnn-output transformed into the world coordinate system.

            Beware, position correction is handled in the evaluation scripts. It's much simpler as we only have
            to consider the offset and scaling due to the cropping and resizing to the CNN input size.

        Args:
            image_size: B x [Width, Height]
        
        Returns:
            Updated rotations.
        '''
        xy_image = coord[..., :2]
        half_image_size_tensor = 0.5 * image_sizes
        xy_normalized = (xy_image - half_image_size_tensor) / half_image_size_tensor[0]
        fs = torch.as_tensor(self.f, device=xy_image.device).expand_as(xy_normalized[..., :-1])
        xyz = torch.cat([xy_normalized, fs], dim=-1)
        m = PerspectiveCorrector._make_look_at_matrix(xyz)
        out = torchquaternion.mult(torchquaternion.from_matrix(m), pose)
        return out

    def _make_look_at_matrix(pos: Tensor):
        '''Computes a rotation matrix where the z axes is aligned with the argument vector.

        This leaves a degree of rotation around the this axis. This is resolved by constraining
        the x axis to the horizonal plane (perpendicular to the global y-axis).
        '''
        z = pos / torch.norm(pos, dim=-1, keepdim=True)
        x = torch.cross(*torch.broadcast_tensors(pos.new_tensor([0.0, 1.0, 0.0]), z), dim=-1)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.cross(z, x, dim=-1)
        y = y / torch.norm(x, dim=-1, keepdim=True)
        M = torch.stack([x, y, z], dim=-1)
        return M


class AlignedRotationErrorMetric(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        error_mode: Literal['euler', 'geo'],
        correction_mode: Literal['perspective', 'opal23'],
        fov: float | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state("image_sizes", default=[], dist_reduce_fx="cat")
        self.add_state("target_quats", default=[], dist_reduce_fx="cat")
        self.add_state("pred_quats", default=[], dist_reduce_fx="cat")
        self.add_state("pred_coord", default=[], dist_reduce_fx="cat")
        self.add_state("individual", default=[], dist_reduce_fx="cat")
        self._correction_mode = correction_mode
        self._fov = fov
        self._error_mode = error_mode

    def update(self, preds: Batch, targets: Batch) -> None:
        self.target_quats.append(targets['pose'])
        self.pred_quats.append(preds['pose'])
        self.pred_coord.append(preds['coord'])
        if self._correction_mode == 'perspective':
            # FIXME: images should come in NCHW format!
            image_sizes = torch.as_tensor([t.shape[-3:-1] for t in targets['image']])
            self.image_sizes.append(image_sizes)  # Format HW
        else:
            self.individual.append(targets['individual'])

    def compute(self) -> Tensor:
        target_quats = torch.cat(self.target_quats)
        pred_quats = torch.cat(self.pred_quats)
        pred_coord = torch.cat(self.pred_coord)
        if self._correction_mode == 'perspective':
            image_sizes = torch.flip(torch.cat(self.image_sizes), dims=(-1,))  # Format to WH
            corrector = PerspectiveCorrector(self._fov)
            pred_quats = corrector.corrected_rotation(image_sizes, pred_coord, pred_quats)
        else:
            individual = torch.cat(self.individual)
            pred_quats = compute_opal_paper_alignment(pred_quats, target_quats, individual)
        if self._error_mode == 'euler':
            return _aflw3d_euler_errors(pred_quats, target_quats)
        else:
            return torchquaternion.geodesicdistance(pred_quats, target_quats)
