import torch
import neuralnets.models
import timeit


def timing_and_output(net, x):
    with torch.no_grad():
        print (f"Input of {tuple(x.shape)} sized image batch into {type(net).__name__}")
        out = net(x)
        print (f"Output is a set of tensors of dimensions {[tuple(t.shape) for t in out]}")
        N = 100
        time = timeit.timeit('net(x)', number=N, globals={ 'net' : net, 'x' : x })
        print (f"Inference time: {time/N*1000:.0f} ms averaged over {N} runs")


if __name__ == '__main__':
    torch.set_num_threads(1)

    net = neuralnets.models.MobilnetV1WithPointHead()
    net.eval()
    timing_and_output(net, torch.rand(1, 1, net.input_resolution, net.input_resolution))

    net = neuralnets.models.LocalizerNet()
    net.eval()
    timing_and_output(net, torch.rand(1, 1, net.input_resolution[0],net.input_resolution[1]))