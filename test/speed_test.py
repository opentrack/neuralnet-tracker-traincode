import sys
import numpy as np
import timeit
import os

if __name__ == '__main__':
    # Only single thread for inference time measurement
    os.environ["OMP_NUM_THREADS"] = "1"
    # Import after setting the environment variable to make the setting actually take effect!
    # See https://github.com/microsoft/onnxruntime/issues/3233
    import onnxruntime as ort

    # In case ONNX RT was not build with OMP support, I control the number of threads the following way
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    #opts.enable_profiling = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    #opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    # This option causes a 60% increased runtime! Why? I mean it would be totally fine if it increased the runtime
    # for the first inference invocation but it makes them all generally slower.
    # Tested with recent onnx runtime v 1.9.0 installed with pip.
    # opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort_session = ort.InferenceSession(sys.argv[1],sess_options=opts)

    img_size = 129

    x = np.zeros((1, 1, img_size, img_size), dtype=np.float32)

    N = 100
    time = timeit.timeit("ort_session.run(None, { 'x': x })", number=N, globals={ 'ort_session' : ort_session, 'x' : x })
    print (f"Inference time: {time/N*1000:.0f} ms averaged over {N} runs")