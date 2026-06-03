from typing import Literal
import torch
import timeit
import onnxruntime as ort
import numpy as np
import copy

import trackertraincode.neuralnets.models
from trackertraincode.neuralnets.rotrepr import RotationRepr
from scripts.export_model import convert_posemodel_onnx


@torch.no_grad()
def run_pytorch_inference(net, x):
    print(f"Input of {tuple(x.shape)} sized image batch into {type(net).__name__}")
    out = net(x)
    if isinstance(out, dict):
        print(f"Output is a dict of tensors of dimensions {[(k,t.shape) for k,t in out.items()]}")
    else:
        print(f"Output is a tuple of tensors of dimensions {[t.shape for t in out]}")


def num_parameters(net: torch.nn.Module):
    nparams = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {nparams / (10**6) : 0.1f}")


@torch.no_grad()
def runtime_pytorch(net, x, device):
    x = x.to(device) if device is not None else x
    net = net.to(device) if device is not None else net
    N = 100
    time = timeit.timeit("net(x)", number=N, globals={"net": net, "x": x})
    print(f"Torch Inference time {device}: {time/N*1000:.0f} ms averaged over {N} runs")


def backprop(net):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 16
    net = copy.deepcopy(net)
    net.train()
    net.to(device)
    torch.autograd.set_detect_anomaly(True)
    out = net(torch.rand(B, 1, net.input_resolution, net.input_resolution, device=device))

    def fake_loss(x: torch.Tensor | RotationRepr):
        if hasattr(x, "value"):
            x = x.value
        return torch.sum(x)

    val = torch.sum(torch.stack([fake_loss(o) for o in out.values()]))
    val.backward()


def compile_compat(net, x):
    net_for_compilation = copy.deepcopy(net)
    del net
    net_for_compilation.train()
    # Compilation is apparently inplace.
    net_for_compilation.compile(fullgraph=True, dynamic=False)
    if torch.cuda.is_available():
        net_for_compilation = net_for_compilation.to("cuda")
        x = x.to("cuda")
    net_for_compilation(x)


def onnx_export_and_inference_speed(net, device: Literal["cpu", "cuda"]):
    modelfilename = "/tmp/net.onnx"
    convert_posemodel_onnx(net, modelfilename, verbose=False)
    providers = {
        "cpu": [
            "CPUExecutionProvider",
        ],
        # This may look odd but ONNX can decide to run certain ops on the CPU!
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }[device]
    ort_session = ort.InferenceSession(modelfilename, providers=providers)
    x = np.random.normal(size=(1, 1, net.input_resolution, net.input_resolution)).astype(np.float32)
    N = 100
    time = timeit.timeit(
        "ort_session.run(None, { 'x' :  x })",
        number=N,
        globals={"ort_session": ort_session, "x": x},
    )
    print(f"ONNX Inference ({device}) time: {time/N*1000:.01f} ms averaged over {N} runs")


def test_pose_network_sanity(tmp_path):
    torch.set_num_threads(1)
    net = trackertraincode.neuralnets.models.NetworkWithPointHead(
        config="resnet18",
        enable_uncertainty=True,
        input_resolution=129,
        backbone_args={"use_blurpool": True},
    )
    net.eval()
    print("==== Pose Net ====")
    num_parameters(net)
    run_pytorch_inference(net, torch.rand(1, 1, net.input_resolution, net.input_resolution))
    print("-------------")
    runtime_pytorch(net, torch.rand(1, 1, net.input_resolution, net.input_resolution), "cuda")
    runtime_pytorch(net, torch.rand(1, 1, net.input_resolution, net.input_resolution), "cpu")
    print("-------------")
    compile_compat(net, torch.rand(1, 1, net.input_resolution, net.input_resolution))

    filename = tmp_path / "model.ckpt"
    trackertraincode.neuralnets.models.save_model(net, filename)
    trackertraincode.neuralnets.models.load_model(filename)

    onnx_export_and_inference_speed(net, "cpu")
    onnx_export_and_inference_speed(net, "cuda")

    # Check if gradients can be computed.
    backprop(net)


def test_localizer_sanity():
    torch.set_num_threads(1)

    net = trackertraincode.neuralnets.models.LocalizerNet()
    net.eval()
    print("---------- LocalizerNet ----------")
    num_parameters(net)
    run_pytorch_inference(net, torch.rand(1, 1, net.input_resolution[0], net.input_resolution[1]))
    print("-------------")
    runtime_pytorch(net, torch.rand(1, 1, net.input_resolution[0], net.input_resolution[1]), "cuda")
    runtime_pytorch(net, torch.rand(1, 1, net.input_resolution[0], net.input_resolution[1]), "cpu")
