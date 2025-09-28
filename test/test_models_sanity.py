from typing import Literal
import torch
import timeit
import onnxruntime as ort
import numpy as np
import copy

import trackertraincode.neuralnets.models
from trackertraincode.neuralnets.rotrepr import RotationRepr
from scripts.export_model import convert_posemodel_onnx


def timing_and_output(net, x):
    with torch.no_grad():
        print (f"Input of {tuple(x.shape)} sized image batch into {type(net).__name__}")
        out = net(x)
        if isinstance(out, dict):
            print (f"Output is a dict of tensors of dimensions {[tuple(t.shape) for t in out.values()]}")
        else:
            print (f"Output is a tuple of tensors of dimensions {[tuple(t.shape) for t in out]}")
        N = 100
        time = timeit.timeit('net(x)', number=N, globals={ 'net' : net, 'x' : x })
        print (f"Torch Inference time: {time/N*1000:.0f} ms averaged over {N} runs")


def backprop(net):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 16
    net = copy.deepcopy(net)
    net.train()
    net.to(device)
    torch.autograd.set_detect_anomaly(True)
    out = net(torch.rand(B, 1, net.input_resolution, net.input_resolution, device=device))
    def fake_loss(x : torch.Tensor | RotationRepr):
        if hasattr(x,'value'):
            x = x.value
        return torch.sum(x)
    val = torch.sum(torch.stack([ fake_loss(o) for o in out.values() ]))
    val.backward()


def onnx_export_and_inference_speed(net, device : Literal['cpu','cuda']):
    modelfilename = '/tmp/net.onnx'
    convert_posemodel_onnx(net, modelfilename)
    providers = {
        'cpu': [
            'CPUExecutionProvider',
        ],
        # This may look odd but ONNX can decide to run certain ops on the CPU!
        'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    }[device]
    ort_session = ort.InferenceSession(modelfilename, providers=providers)
    x = np.random.normal(size=(1,1,net.input_resolution,net.input_resolution)).astype(np.float32)
    N = 100
    time = timeit.timeit("ort_session.run(None, { 'x' :  x })", number=N, globals={ 'ort_session' : ort_session, 'x' : x })
    print (f"ONNX Inference ({device}) time: {time/N*1000:.01f} ms averaged over {N} runs")


def test_pose_network_sanity(tmp_path):
    torch.set_num_threads(1)
    net = trackertraincode.neuralnets.models.NetworkWithPointHead(config='resnet18', enable_uncertainty=True)
    net.eval()
    print ("---------- Pose Net ----------")
    timing_and_output(net, torch.rand(1, 1, net.input_resolution, net.input_resolution))

    filename = tmp_path / 'model.onnx'
    trackertraincode.neuralnets.models.save_model(net, filename)
    trackertraincode.neuralnets.models.load_model(filename)

    onnx_export_and_inference_speed(net, 'cpu')
    onnx_export_and_inference_speed(net, 'cuda')

    # Check if gradients can be computed.
    backprop(net)


def test_localizer_sanity():
    torch.set_num_threads(1)

    net = trackertraincode.neuralnets.models.LocalizerNet()
    net.eval()
    print ("---------- LocalizerNet ----------")
    timing_and_output(net, torch.rand(1, 1, net.input_resolution[0],net.input_resolution[1]))