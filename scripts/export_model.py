from os.path import splitext
import argparse
import numpy as np
import os
import torch
import copy
import tqdm
import itertools

from torch.ao.quantization import get_default_qconfig_mapping, QConfig, QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.ao.quantization import fake_quantize
from torch.ao.quantization import observer

import torch.onnx
import torch.nn as nn
import onnx
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print ("Warning cannot import ONNX runtime: Runtime checks disabled")

from trackertraincode.neuralnets.bnfusion import fuse_convbn
import trackertraincode.neuralnets.models
import trackertraincode.pipelines

# Only single thread for inference time measurement
os.environ["OMP_NUM_THREADS"] = "1"



def clear_denormals(state_dict, threshold=1.e-20):
    # I tuned the threshold so I don't see a performance
    # decrease compared to pretrained weights from torchvision.
    # The real denormals start below 2.*10^-38
    # Denormals make computations on CPU very slow .. at least back then ...
    state_dict = { k:v.detach().clone() for k,v in state_dict.items() }
    print ("Denormals or zeros:")
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            mask = torch.abs(v) > threshold
            n = torch.count_nonzero(~mask)
            if n:
                print (f"{k:40s}: {n:10d} ({n/np.prod(v.shape)*100}%)")
            v *= mask.to(torch.float32)
    return state_dict



def quantize_backbone(original : trackertraincode.neuralnets.models.NetworkWithPointHead):
    original = copy.deepcopy(original)

    dsid = trackertraincode.pipelines.Id
    train_loader, _, _ = trackertraincode.pipelines.make_pose_estimation_loaders(
        inputsize = original.input_resolution, 
        batchsize = 128,
        datasets = [dsid.REPO_300WLP,dsid.SYNFACE,dsid],
        dataset_weights = {},
        use_weights_as_sampling_frequency=True,
        enable_image_aug=True,
        rotation_aug_angle=30.,
        roi_override='original',
        device=None)
    example_input = (next(iter(train_loader))[0]['image'],)
    original.eval()

    # Configuration chosen as per advice from 
    # https://oscar-savolainen.medium.com/how-to-quantize-a-neural-network-model-in-pytorch-an-in-depth-explanation-d4a2cdf632a4
    config = QConfig(activation=fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(observer=observer.MovingAverageMinMaxObserver,
                                                                                    quant_min=0,
                                                                                    quant_max=255,
                                                                                    dtype=torch.quint8,
                                                                                    qscheme=torch.per_tensor_affine),
                                weight=fake_quantize.FakeQuantize.with_args(observer=fake_quantize.MovingAveragePerChannelMinMaxObserver,
                                                                                  quant_min=-128,
                                                                                  quant_max=127,
                                                                                  dtype=torch.qint8,
                                                                                  qscheme=torch.per_channel_symmetric))
    #qconfig_mapping = get_default_qconfig_mapping("x86")
    #qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    #qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping = QConfigMapping()
    qconfig_mapping.set_global(config)
    # Disable quantization after the convolutional layers.
    # The final relu+global pooling seems to be fast enough to do in float32 without
    # significant slowdown.
    qconfig_mapping = qconfig_mapping.set_object_type(nn.AdaptiveAvgPool2d, None)
    if original.config == 'resnet18':
        # TODO: better SW design?
        # Disables quantization of the input of the AveragePooling
        qconfig_mapping = qconfig_mapping.set_module_name('layers.7.1.relu', None)

    convnet = prepare_fx(
        fuse_convbn(torch.fx.symbolic_trace(original.convnet)), 
        qconfig_mapping, 
        example_input)

    for _, batches in tqdm.tqdm(zip(range(20), train_loader)):
        for batch in batches:
            convnet(batch['image'])

    original.convnet = convert_fx(convnet)

    return original


class ModelForOpenTrack(nn.Module):
    """
        Rearranges the model output
    """
    def __init__(self, original : trackertraincode.neuralnets.models.NetworkWithPointHead):
        super(ModelForOpenTrack,self).__init__()
        self._original = original
        self.input_names = ['x']
        self._output_name_map = [
            ('coord' , 'pos_size'),
            ('pose'  , 'quat'    ),
            ('roi'   , 'box'     ),
        ]
        if original.enable_uncertainty:
            self._output_name_map += [
                ('coord_scales'     , 'pos_size_scales_tril'),
                ('pose_scales_tril' , 'rotaxis_scales_tril'),
                ('roi_scales'       , 'box_scales'),
            ]
    
    @property
    def output_names(self):
        return [ n for _,n in self._output_name_map ]

    @property
    def input_resolution(self):
        return self._original.input_resolution

    def forward(self, x):
        y = self._original(x)
        return tuple(y[k] for k,_ in self._output_name_map)


class ExportModel(nn.Module):
    def __init__(self, original : nn.Module):
        super().__init__()
        self._original = original
        self.input_names = ['x']
        self.output_names = ExportModel._compute_output_names(original)

    @staticmethod
    def _compute_output_names(original):
        original.eval()
        x = torch.zeros((1,1,original.input_resolution, original.input_resolution))
        y = original(x)
        return list(y.keys())

    @property
    def input_resolution(self):
        return self._original.input_resolution

    def forward(self, x):
        y = self._original(x)
        return tuple(y[k] for k in self.output_names)


def load_facelocalizer(args):
    sd = torch.load(args.localizermodelfilename)
    net = trackertraincode.neuralnets.models.LocalizerNet()
    net.load_state_dict(sd, strict=True)
    return net


def print_io_info(ort_session):
    print ("Inputs:")
    for node in ort_session.get_inputs():
        print (f"\t{node.name}, {node.shape}, {node.type}")
    print ("Outputs:")
    for node in ort_session.get_outputs():
        print (f"\t{node.name}, {node.shape}, {node.type}")


def compare_network_outputs(torchmodel, ort_session, inputs):
    torch_out = torchmodel(*inputs)
    outputs = ort_session.run(None, {
        k:v.detach().numpy() for k,v in zip(torchmodel.input_names, inputs)
    })
    # Outputs better be the same ...
    for a, b, name in zip(torch_out, outputs, torchmodel.output_names):
        a = a.detach().numpy()
        if not np.allclose(a, b, 1.e-4):
            delta = np.amax(np.abs(a - b))
            print(f"WARNING Torch output of {name}, differs from Onnx output by max {delta}")


@torch.no_grad()
def convert_posemodel_onnx(net : nn.Module, filename, for_opentrack=True, quantize=False):
    net.load_state_dict(clear_denormals(net.state_dict()))
    if quantize:
        net = quantize_backbone(net)
    if for_opentrack:
        net = ModelForOpenTrack(net)
    else:
        net = ExportModel(net)
    net.eval()

    # Batchsize. For opentrack we need 1. Otherwise just pick something larger to see that it works more generally.
    B = 1 if for_opentrack else 5
    # WARNING: Inputs must be tuple. List won't work.
    inputs = (torch.randn(
        B, 1, net.input_resolution, net.input_resolution),)

    B, C, H, W = inputs[0].shape

    destination = splitext(filename)[0]+('_ptq' if quantize else '')+('.onnx' if for_opentrack else '_complete.onnx')

    print (f"Exporting {net.__class__}, input size = {H},{W} to {destination}")

    dynamic_axes = None if for_opentrack else \
       { k:{0 : 'batch' } for k in (net.input_names+net.output_names) }

    torch.onnx.export(
        net,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        destination,  # where to save the model (can be a file or file-like object)
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        keep_initializers_as_inputs=False,
        input_names = net.input_names,
        output_names = net.output_names,
        dynamic_axes = dynamic_axes,
        verbose=False)

    # torch.onnx.dynamo_export(
    #     net,  # model being run
    #     inputs,  # model input (or a tuple for multiple inputs)
    #     export_options=torch.onnx.ExportOptions(
    #         dynamic_shapes=False,
    #     )).save(destination)

    onnxmodel = onnx.load(destination)
    onnxmodel.doc_string = 'Head pose prediction'
    onnxmodel.model_version = 4  # This must be an integer or long.
    
    onnx.checker.check_model(onnxmodel)
    
    onnx.save(onnxmodel, destination)

    if ort is not None:
        ort_session = ort.InferenceSession(
            onnxmodel.SerializeToString(), 
            providers=['CPUExecutionProvider'])
        print_io_info(ort_session)
        compare_network_outputs(net, ort_session, inputs)


def convert_localizer(args):
    net = load_facelocalizer(args)
    net.load_state_dict(clear_denormals(net.state_dict()))
    net.eval()
    H, W = net.input_resolution[0], net.input_resolution[1]
    destination = splitext(args.localizermodelfilename)+'.onnx'

    x = torch.randn(1, 1, H, W, requires_grad=False)
    torch_out = net(x)

    torch.onnx.export(
        net,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        destination,   # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['x'],
        output_names = ['logit_box'],
        verbose=False)

    del net

    onnxmodel = onnx.load(destination)
    onnx.checker.check_model(onnxmodel)
    del onnxmodel

    if ort is not None:
        ort_session = ort.InferenceSession(destination)
        outputs = ort_session.run(None, {
            'x': x.detach().numpy()
        })
        for a, b in zip(torch_out, outputs):
            if not np.allclose(a, b, 1.e-4):
                print(f"WARNING Torch output: {a} differs from Onnx output {b}")
        del ort_session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert networks to onnx format")
    parser.add_argument('--posenet', dest = 'posemodelfilename', help="filename of model checkpoint", type=str, default=None)
    parser.add_argument('--full', action='store_true', default=False)
    parser.add_argument('--localizer', dest = 'localizermodelfilename', help="filename of model checkpoint", type=str, default=None)
    parser.add_argument('--quantize', action='store_true', default=False)
    args = parser.parse_args()
    if args.posemodelfilename:
        net = trackertraincode.neuralnets.models.load_model(args.posemodelfilename) 
        convert_posemodel_onnx(net, args.posemodelfilename, for_opentrack=not args.full, quantize=args.quantize)
    if args.localizermodelfilename:
       convert_localizer(args)
    if not args.posemodelfilename and not args.localizermodelfilename:
        print ("No input models. No action taken.")