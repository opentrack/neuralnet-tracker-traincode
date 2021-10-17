import torch
import neuralnets.models
import timeit


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


def test_neuralnetwork_model_sanity():
    torch.set_num_threads(1)

    net = neuralnets.models.LocalAttentionNetwork()
    net.eval()
    print (net)
    timing_and_output(net, torch.rand(1, 1, net.input_resolution, net.input_resolution))

    # Check if gradients can be computed.
    net = neuralnets.models.LocalAttentionNetwork()
    net.train()
    torch.autograd.set_detect_anomaly(True)
    out = net(torch.rand(3, 1, net.input_resolution, net.input_resolution))
    val = torch.sum(torch.cat([ torch.sum(o)[None] for o in out.values() ]))
    val.backward()

    net = neuralnets.models.LocalizerNet()
    net.eval()
    timing_and_output(net, torch.rand(1, 1, net.input_resolution[0],net.input_resolution[1]))


if __name__ == '__main__':
    test_neuralnetwork_model_sanity()