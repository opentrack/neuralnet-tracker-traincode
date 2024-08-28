import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple, Dict, Any, Union, List, NamedTuple
from copy import copy
from numpy.typing import NDArray

import torch
from torch import nn
from torch import Tensor

from trackertraincode.datasets.batch import Batch, Metadata
from trackertraincode.neuralnets.affine2d import Affine2d

import trackertraincode.datatransformation as dtr
import trackertraincode.utils as utils
from trackertraincode.pipelines import whiten_image

from abc import ABCMeta, abstractmethod


class InferenceNetwork(metaclass = ABCMeta):
    @abstractmethod
    def __call__(self, images : Tensor) -> Dict[str, Tensor]:
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
        return session.run(None, {
            'x': img[None,...]
        })
    outputs = zip(*(_infer_one(img) for img in batch.numpy()))
    outputs = [ np.vstack(o) for o in outputs ]
    return outputs


def _onnx_batch_inference(session, batch):
    return session.run(None, {
        'x': batch.numpy()
    })


def load_pose_network(filename, device) -> InferenceNetwork:
    if filename.endswith('.onnx'):
        import onnxruntime
        # For showing device placement
        #onnxruntime.set_default_logger_severity(1)
        #onnxruntime.set_default_logger_verbosity(0)

        class OnnxPoseNetwork(InferenceNetwork):
            def __init__(self, modelfile, device):
                providers = {
                    'cpu' :  [ 'CPUExecutionProvider', ],
                    # This may look odd but ONNX can decide to run certain ops on the CPU!
                    'cuda' : [ 'CUDAExecutionProvider', 'CPUExecutionProvider' ]
                }[device]
                self.session = onnxruntime.InferenceSession(modelfile, providers=providers)
                # TODO: decide on a names. Also for the OpenTrack plugin.
                namemap = {
                    'pos_size' : 'coord',
                    'quat' : 'pose',
                    'box' : 'roi',
                    'eyes' : 'eyeparam', 
                    'pos_size_scales'  : 'coord_scales',
                    'pos_size_std' : 'coord_scales',
                    'rotaxis_scales_tril' : 'pose_scales_tril',
                    'rotaxis_std' : 'pose_scales_tril',
                    'rot_conc_tril' : 'pose_conc_tril',
                    'box_scales' : 'roi_scales',
                    'box_std' : 'roi_scales'
                }
                self.output_names = [
                    namemap.get(o.name,o.name) for o in self.session.get_outputs() ]
                #print ("Model version string: ", self.session.get_modelmeta().version, ' has outputs', self.output_names)
                
                if isinstance(self.session.get_inputs()[0].shape[0],int):
                    self._inference = _onnx_single_frame_inference
                else:
                    self._inference = _onnx_batch_inference


            @property
            def device_for_input(self):
                '''In spite of potentially executing on the gpu, ONNX does not know what a pytorch Tensor on the GPU is.'''
                return 'cpu'

            @property
            def input_resolution(self) -> int:
                return 129 # TODO: this should somehow come from the model file

            def __call__(self, batch):
                outputs = self._inference(self.session, batch)
                outputs = dict(zip(self.output_names, outputs))
                if self.session.get_modelmeta().version not in (2,3,4):
                    # Old model needs coordinate conversion
                    quats = outputs['pose']
                    x, y, z = quats[...,0].copy(), quats[...,1].copy(), quats[...,2].copy()
                    quats[...,0] = -z
                    quats[...,1] = -y
                    quats[...,2] = -x
                    outputs['pose'] = quats
                outputs = { k:torch.from_numpy(v) for k,v in outputs.items() }
                return outputs
                
        return OnnxPoseNetwork(filename, device)
    else:
        import trackertraincode.neuralnets.models

        class PytorchPoseNetwork(InferenceNetwork):
            def __init__(self, modelfile, device):
                # The checkpoints are not self-describing. So we must have a matching network in code.
                net = trackertraincode.neuralnets.models.NetworkWithPointHead(
                    enable_point_head=True, 
                    enable_face_detector=False, 
                    config='mobilenetv1', 
                    enable_uncertainty=True)
                state_dict = torch.load(modelfile)
                net.load_state_dict(state_dict)
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


def _apply_backtrafo(backtrafo : Affine2d, batch : Batch):
    batch = copy(batch)
    for k, v in batch.items():
        c = dtr.get_category(batch,k)
        if c in dtr.imagelike_categories:
            continue
        batch[k] = dtr.apply_affine2d(backtrafo, k, v, c)
    return batch


@torch.no_grad()
def predict(net : InferenceNetwork, images : List[Tensor], rois : Optional[Tensor] = None, focus_roi_expansion_factor : float = 1.2) -> Batch:
    '''
    Args:
        net : Inference function
        images : Unnormalized uint8 images in HWC format
        rois: Tensor in x0,y0,x1,y1 format
        focus_roi_expansion_factor: Factor by which to enlarge the cropping region. Normally it's the squareified ROI.
    '''
    B = len(images)
    H,W,C = images[-1].shape
    input_device = images[-1].device
    assert rois is None or rois.device == input_device

    roi_focus = None
    if rois is not None:
        assert rois.shape == (B,4)
        roi_focus = dtr.FocusRoi(net.input_resolution, focus_roi_expansion_factor, insert_backtransform=True)
    else:
        rois = [ None for _ in range(B) ]

    def create_batch(image, roi):
        image = dtr._ensure_image_nchw(dtr.from_numpy_or_tensor(image))
        b = Batch(Metadata((H,W), 0, categories={'image' : dtr.FieldCategory.image}), {'image' : image })
        if roi is not None:
            b['roi'] = dtr.from_numpy_or_tensor(roi)
            b.meta.categories['roi'] = dtr.FieldCategory.roi
        if roi_focus is not None:
            b = roi_focus(b)
        return b
    
    batch = [ create_batch(i,r) for i,r in zip(images, rois) ]
    batch = Batch.collate(batch)

    batch = dtr.normalize_batch(batch)

    preds = net(whiten_image(batch['image']).to(net.device_for_input))
    
    del batch['image'] # Save some resources

    preds = Batch(batch.meta, **preds)
    preds.meta.categories.update({
        'coord' : dtr.FieldCategory.xys,
        'pose' : dtr.FieldCategory.quat,
        'pt3d_68' : dtr.FieldCategory.points,
    })
    preds = dtr.unnormalize_batch(preds)

    if net.device_for_input != input_device:
        preds = dtr.to_device(input_device, preds)

    if roi_focus is not None:
        batch = dtr.unnormalize_batch(batch)
        backtrafo = Affine2d(batch['image_backtransform'])
        preds = _apply_backtrafo(backtrafo, preds)
    return preds



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
        mask = (sample['hasface']>self.threshold)
        mask &= (pred['hasface']>self.threshold)
        err = F.mse_loss(pred['roi'], target[:,:], reduction='none')
        err[~mask,:] = np.nan
        err0 = torch.sum(err[:,:2], dim=1)
        err1 = torch.sum(err[:,2:], dim=1)
        return torch.cat([err0[:,None], err1[:,None]],dim=1)


class LocalizerIsFaceMatches(object):
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, pred, sample):
        target = sample['hasface']
        score = pred['hasface']
        match = torch.eq(target>self.threshold, score>self.threshold)
        return match


class PoseErr(object):
    c = nn.MSELoss(reduction='none')
    def __call__(self, pred, batch):
        assert isinstance(pred, dict)
        coord_target = batch['coord'].cpu().numpy()
        quat_target = batch['pose'].cpu().numpy()
        x0, y0, x1, y1 = np.moveaxis(batch['roi'].cpu().numpy(), -1, 0)
        coord = pred['coord']
        quat  = pred['pose']
        coord = coord.cpu().numpy()
        quat  = quat.cpu().numpy()
        coord_errs = np.abs(coord-coord_target) / (x1-x0)[:,None]
        rot_errs = (utils.convert_to_rot(quat_target).inv()*utils.convert_to_rot(quat)).magnitude()
        result = np.concatenate([rot_errs[:,None], coord_errs], axis=1)
        return result


class EulerAngleErrors(object):
    def __call__(self, pred, batch):
        quat_target = batch['pose'].cpu().numpy()
        quat  = pred['pose'].cpu().numpy()
        euler_target = np.array([ utils.inv_aflw_rotation_conversion(q) for q in utils.convert_to_rot(quat_target) ])
        euler = np.array([ utils.inv_aflw_rotation_conversion(q) for q in utils.convert_to_rot(quat) ])
        errors = utils.angle_errors(euler_target, euler)
        return errors


def _eval_keypoints(pred : Tensor, gt : Tensor, dims = 3):
    # Pred and GT are bached point tensors with shapes
    # B x N x 3, where N is the number of keypoints
    B, N, D = pred.shape
    assert D == 3
    assert pred.shape == gt.shape
    pred = pred.clone()
    gt = gt.clone()
    pred[:, :, 2] -= torch.mean(pred[:, :, 2], dim=-1, keepdim=True)
    gt  [:, :, 2] -= torch.mean(gt  [:, :, 2], dim=-1, keepdim=True)
    dist   = torch.mean(torch.norm(pred[:,:,:dims] - gt[:,:,:dims], dim=-1), dim=-1)
    left   = torch.amin(gt[:, :, 0], dim=1)
    right  = torch.amax(gt[:, :, 0], dim=1)
    top    = torch.amin(gt[:, :, 1], dim=1)
    bottom = torch.amax(gt[:, :, 1], dim=1)
    bbox_size = torch.sqrt((right - left) * (bottom - top))
    dist = dist / bbox_size
    return dist.cpu().numpy()


# Adapted from https://github.com/MCG-NJU/SADRNet/blob/main/src/model/loss.py#L224
class UnweightedKptNME:
    def __init__(self, dimensions=3):
        self.dims = dimensions

    def __call__(self, pred, batch):
        return _eval_keypoints(pred['pt3d_68'], batch['pt3d_68'], self.dims)


class KptNmeResults(NamedTuple):
    bin_30_nme : float
    bin_60_nme : float
    bin_90_nme : float
    avg_nme    : float


class KptNME:
    def __init__(self, dimensions=3):
        self.dims = dimensions
    
    def __call__(self, pred, batch):
        #return self._eval_keypoints(pred['pt3d_68'], batch['pt3d_68'])
        masks = self._compute_bin_masks(batch['pose'])
        nme_by_bins = [ np.average(_eval_keypoints(pred['pt3d_68'][m], batch['pt3d_68'][m], self.dims), axis=0).tolist() for m in masks ]
        return KptNmeResults(
            *nme_by_bins,
            np.average(nme_by_bins).tolist()
        )

    def _compute_bin_masks(self, pose_gt : Tensor):
        '''Masks for the yaw bins from the literature: 0-30, 30-60, 60-90 deg.'''
        rot = utils.convert_to_rot(pose_gt.cpu().numpy())
        pyr_gt = np.array([ utils.inv_aflw_rotation_conversion(r) for r in rot ])
        abs_yaw_deg = np.abs(pyr_gt[:,1]) * 180./np.pi
        bounds_list = [(0.,30.),(30.,60.),(60.,90.)]
        masks = [
            ((a <= abs_yaw_deg) & (abs_yaw_deg < b)) for (a,b) in bounds_list
        ]
        return masks
