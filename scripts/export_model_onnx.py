from os.path import splitext
import argparse

import torch.onnx
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import neuralnets.models
import timeit
import os

# Only single thread for inference time measurement
os.environ["OMP_NUM_THREADS"] = "1"

class Modelwrapper(nn.Module):
    """
        Rearranges the model output
    """
    def __init__(self, original, args):
        super(Modelwrapper,self).__init__()
        self.original = original
        self.output_names = ['pos_size','quat','box']
        if args.exportall:
            self.output_names += ['shapeparams']
            self.output_names += ['pt3d_68']

    @property
    def input_resolution(self): return self.original.input_resolution

    def forward(self, x):
        y = self.original(x)
        out = y['coord'], y['pose'], y['roi']
        if 'pt3d_68' in self.output_names:
            out = out + (y['pt3d_68'],)
        if 'shapeparams' in self.output_names:
            out = out + (y['shapeparams'],)
        return out
            


def load_posemodel(args):
    sd = torch.load(args.posemodelfilename)
    neuralnets.models.clear_denormals_inplace(sd)
    #net = neuralnets.models.NetworkWithPointHead(enable_point_head=False, enable_face_detector=True, enable_full_head_box=True)
    net = neuralnets.models.LocalAttentionNetwork(enable_face_detector=True, enable_full_head_box=True)
    net.load_state_dict(sd, strict=True)
    net.eval()
    net = Modelwrapper(net, args)
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
    destination = splitext(args.posemodelfilename)[0]+'.onnx'

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
        output_names = net.output_names,
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

    # Speet test
    x = x.detach().numpy()
    N = 100
    time = timeit.timeit("ort_session.run(None, { 'x': x })", number=N, globals={ 'ort_session' : ort_session, 'x' : x })
    print (f"Inference time: {time/N*1000:.0f} ms averaged over {N} runs")
    # Should yield ca 5 ms (MobileNetV1x0.75). For comparison cleardusks 3DDFA_V2, in the standard MobileNetV1 variant runs in 8ms.

    del ort_session


def convert_localizer(args):
    net = load_facelocalizer(args)
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
    ort_session = ort.InferenceSession(destination)
    outputs = ort_session.run(None, {
        'x': x.detach().numpy()
    })

    for a, b in zip(torch_out, outputs):
        assert np.allclose(a.detach().numpy(), b)

    del ort_session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert networks to onnx format")
    parser.add_argument('--posenet', dest = 'posemodelfilename', help="filename of model checkpoint", type=str, default=None)
    parser.add_argument('--localizer', dest = 'localizermodelfilename', help="filename of model checkpoint", type=str, default=None)
    parser.add_argument('--export-all', dest = 'exportall', help='export the model with inference of all available output channels. Otherwise only pose and bounding box', default=False, action='store_true')
    args = parser.parse_args()
    if args.posemodelfilename:
        convert_posemodel(args)
    if args.localizermodelfilename:
       convert_localizer(args)
    if not args.posemodelfilename and not args.localizermodelfilename:
        print ("No input models. No action taken.")
