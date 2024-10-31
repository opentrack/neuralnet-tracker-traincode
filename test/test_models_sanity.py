import torch
import timeit
import onnxruntime as ort
import numpy as np
import copy

import trackertraincode.neuralnets.models
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
        print (f"Inference time: {time/N*1000:.0f} ms averaged over {N} runs")


def backprop(net):
    device = 'cuda'
    B = 16
    net = copy.deepcopy(net)
    net.train()
    net.to(device)
    torch.autograd.set_detect_anomaly(True)
    out = net(torch.rand(B, 1, net.input_resolution, net.input_resolution, device=device))
    val = torch.sum(torch.cat([ torch.sum(o)[None] for o in out.values() ]))
    val.backward()


def onnx_export_and_inference_speed(net):
    modelfilename = '/tmp/net.onnx'
    convert_posemodel_onnx(net, modelfilename)
    ort_session = ort.InferenceSession(modelfilename, providers=['CPUExecutionProvider'])
    x = np.random.normal(size=(1,1,net.input_resolution,net.input_resolution)).astype(np.float32)
    N = 100
    time = timeit.timeit("ort_session.run(None, { 'x' :  x })", number=N, globals={ 'ort_session' : ort_session, 'x' : x })
    print (f"ONNX Inference time: {time/N*1000:.01f} ms averaged over {N} runs")


def test_pose_network_sanity(tmp_path):
    torch.set_num_threads(1)
    net = trackertraincode.neuralnets.models.NetworkWithPointHead(config='mobilenetv1', enable_uncertainty=True)
    net.eval()
    
    timing_and_output(net, torch.rand(1, 1, net.input_resolution, net.input_resolution))

    filename = tmp_path / 'model.onnx'
    trackertraincode.neuralnets.models.save_model(net, filename)
    trackertraincode.neuralnets.models.load_model(filename)

    onnx_export_and_inference_speed(net)

    # Check if gradients can be computed.
    backprop(net)


def test_localizer_sanity():
    torch.set_num_threads(1)

    net = trackertraincode.neuralnets.models.LocalizerNet()
    net.eval()
    timing_and_output(net, torch.rand(1, 1, net.input_resolution[0],net.input_resolution[1]))


if __name__ == '__main__':
    test_pose_network_sanity()
    #test_recurrent_pose_model_sanity() # FIXME: The model is broken