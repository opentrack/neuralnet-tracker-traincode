from os.path import splitext
import argparse
import numpy as np
import os
import torch
import copy

import torch.onnx
import torch.nn as nn
import onnx
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print ("Warning cannot import ONNX runtime: Runtime checks disabled")

import trackertraincode.neuralnets.models

# Only single thread for inference time measurement
os.environ["OMP_NUM_THREADS"] = "1"



def clear_denormals(state_dict, threshold=1.e-20):
    # I tuned the threshold so I don't see a performance
    # decrease compared to pretrained weights from torchvision.
    # The real denormals start below 2.*10^-38
    state_dict = copy.deepcopy(state_dict)
    print ("Denormals or zeros:")
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            mask = torch.abs(v) > threshold
            n = torch.count_nonzero(~mask)
            if n:
                print (f"{k:40s}: {n:10d} ({n/np.product(v.shape)*100}%)")
            v *= mask.to(torch.float32)
    return state_dict


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
                ('coord_scales'     , 'pos_size_scales'),
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


def load_posemodel(args):
    sd = torch.load(args.posemodelfilename)
    net = trackertraincode.neuralnets.models.NetworkWithPointHead(
        enable_point_head=True,
        enable_face_detector=False,
        config='mobilenetv1',
        enable_uncertainty=True
    )
    net.load_state_dict(sd, strict=True)
    return net


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


def compare_network_outputs(torchmodel, ort_session : ort.InferenceSession, inputs):
    torch_out = torchmodel(*inputs)
    outputs = ort_session.run(None, {
        k:v.detach().numpy() for k,v in zip(torchmodel.input_names, inputs)
    })
    # Outputs better be the same ...
    for a, b, name in zip(torch_out, outputs, torchmodel.input_names):
        a = a.detach().numpy()
        if not np.allclose(a, b, 1.e-4):
            delta = np.amax(np.abs(a - b))
            print(f"WARNING Torch output of {name}, differs from Onnx output by max {delta}")


@torch.no_grad()
def convert_posemodel_onnx(net : nn.Module, filename, for_opentrack=True):
    net.load_state_dict(clear_denormals(net.state_dict()))
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

    destination = splitext(filename)[0]+('.onnx' if for_opentrack else '_complete.onnx')

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

    onnxmodel = onnx.load(destination)
    if for_opentrack:
        onnxmodel.doc_string = 'Head pose prediction'
        onnxmodel.model_version = 3  # This must be an integer or long.
    
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
    args = parser.parse_args()
    if args.posemodelfilename:
        net = load_posemodel(args) 
        convert_posemodel_onnx(net, args.posemodelfilename, for_opentrack=not args.full)
    if args.localizermodelfilename:
       convert_localizer(args)
    if not args.posemodelfilename and not args.localizermodelfilename:
        print ("No input models. No action taken.")