from os.path import join, dirname
import argparse

import torch.onnx
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import neuralnets.models


class Modelwrapper(nn.Module):
    """
        Rearranges the model output
    """
    def __init__(self, original):
        super(Modelwrapper,self).__init__()
        self.original = original

    @property
    def input_resolution(self): return self.original.input_resolution

    def forward(self, x):
        y = self.original.inference(x)
        return y['coord'], y['pose'], y['roi']


def load_posemodel(args):
    sd = torch.load(args.posemodelfilename)
    neuralnets.models.clear_denormals_inplace(sd)
    net = neuralnets.models.NetworkWithPointHead()
    net.load_state_dict(sd, strict=True)
    net.eval()
    net = Modelwrapper(net)
    # Note: Cannot do load_state_dict on the wrapper. That'll fail.
    return net


def load_facelocalizer(args):
    sd = torch.load(args.localizermodelfilename)
    neuralnets.models.clear_denormals_inplace(sd)
    net = neuralnets.models.LocalizerNet()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def convert_posemodel(args):
    net = load_posemodel(args)
    H = W = net.input_resolution
    destination = join(args.outdir,'head-pose.onnx')

    x = torch.randn(1, 1, H, W, requires_grad=False)
    torch_out = net(x)

    torch.onnx.export(
        net,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        destination,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['x'],
        output_names = ['pos_size','quat','box'],
        verbose=False)

    del net

    onnxmodel = onnx.load(destination)
    onnx.checker.check_model(onnxmodel)
    del onnxmodel
    ort_session = ort.InferenceSession(destination)
    outputs = ort_session.run(None, {
        'x': x.detach().numpy()
    })

    # Outputs better be the same ...
    for a, b in zip(torch_out, outputs):
        a = a.detach().numpy()
        if not np.allclose(a, b, 1.e-4):
            print(f"Torch output: {a} differs from Onnx output {b}")
        assert np.allclose(a, b, 1.e-4)
    del ort_session


def convert_localizer(args):
    net = load_facelocalizer(args)
    H, W = net.input_resolution[0], net.input_resolution[1]
    destination = join(args.outdir,'head-localizer.onnx')

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
    ort_session = ort.InferenceSession(destination)
    outputs = ort_session.run(None, {
        'x': x.detach().numpy()
    })

    for a, b in zip(torch_out, outputs):
        assert np.allclose(a.detach().numpy(), b)

    del ort_session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert networks to onnx format")
    parser.add_argument('outdir', help='output directory', type=str)
    parser.add_argument('--posenet', dest = 'posemodelfilename', help="filename of model checkpoint", type=str, default=None)
    parser.add_argument('--localizer', dest = 'localizermodelfilename', help="filename of model checkpoint", type=str, default=None)
    args = parser.parse_args()
    if args.posemodelfilename:
        convert_posemodel(args)
    if args.localizermodelfilename:
       convert_localizer(args)
    if not args.posemodelfilename and not args.localizermodelfilename:
        print ("No input models. No action taken.")
