#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <string>
#include <memory>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#ifdef _MSC_VER
#include <windows.h>
#endif

// This is a very quick and dirty measurement of the pose model inference speed.

// How to build on Linux
// There is no makefile yet. So for now simply invoke the compiler directly. Adjust paths as required.
// g++ ./speed_test.cpp -O2 --std=c++17 -o speed_test -I/opt/onnxruntime-linux-x64/include/ -L/opt/onnxruntime-linux-x64/lib/ -lonnxruntime -fopenmp

// How to build on Windows
// Use the project. Adjust include and link paths.

// Run with model filename as first argument
// LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64/lib ./speed_test <path to model.onnx>

int main(int argc, const char** argv)
{
#ifdef _MSC_VER
    wchar_t buffer[4096];
    memset(buffer, 0, sizeof(wchar_t) * 4096);
    ::MultiByteToWideChar(CP_UTF8, 0, argv[1], strlen(argv[1]), buffer, 4096);
    const std::wstring model_path = buffer;
#else
    const std::string model_path = argv[1];
#endif
    int num_threads = 1;

    Ort::Value input_val{nullptr};
    Ort::Value output_val[3] = {
        Ort::Value{nullptr}, 
        Ort::Value{nullptr}, 
        Ort::Value{nullptr}};
    std::array<float, 3> output_coord{};
    std::array<float, 4> output_quat{};
    std::array<float, 4> output_box{};

    Ort::Session session{nullptr};
    Ort::Env env{nullptr};
    Ort::MemoryInfo allocator_info{nullptr};

    env = Ort::Env{
        OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
        "tracker-neuralnet"
    };
    auto opts = Ort::SessionOptions{};

    // Do thread settings here do anything?
    // There is a warning which says to control number of threads via
    // openmp settings. Which is what we do. omp_set_num_threads directly
    // before running the inference pass.

    opts.SetIntraOpNumThreads(num_threads);
    opts.SetInterOpNumThreads(num_threads);
    opts.SetExecutionMode(ORT_SEQUENTIAL);
    //opts.EnableProfiling(L"profile.json");
    // The optimization options incurs a drastic performance penalty. See Python speed_test.
    // But only on Linux. On windows it's always slow :-(
    //opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    //opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    //opts.EnableCpuMemArena();
    allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    session = Ort::Session{env, model_path.c_str(), opts};

    int img_size =  129;
    int n = img_size*img_size;
    std::unique_ptr<float[]> data = std::make_unique<float[]>(n);
    memset(data.get(), 0, sizeof(float)*n);

    {
        const std::int64_t input_shape[4] = { 1, 1, img_size, img_size };
        input_val = Ort::Value::CreateTensor<float>(allocator_info, data.get(), n, input_shape, 4);
    }

    {
        const std::int64_t output_shape[2] = { 1, 3 };
        output_val[0] = Ort::Value::CreateTensor<float>(
            allocator_info, &output_coord[0], output_coord.size(), output_shape, 2);
    }

    {
        const std::int64_t output_shape[2] = { 1, 4 };
        output_val[1] = Ort::Value::CreateTensor<float>(
            allocator_info, &output_quat[0], output_quat.size(), output_shape, 2);
    }

    {
        const std::int64_t output_shape[2] = { 1, 4 };
        output_val[2] = Ort::Value::CreateTensor<float>(
            allocator_info, &output_box[0], output_box.size(), output_shape, 2);
    }

    const char* input_names[] = {"x"};
    const char* output_names[] = {"pos_size", "quat", "box"};

    const auto nt = omp_get_num_threads();
    omp_set_num_threads(num_threads);
    for (int i=0; i<10; ++i)
    {
        auto t_ = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_val, 1, output_names, output_val, 3);
        std::cout << "time " << i << " = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_).count() << std::endl;
    }
    omp_set_num_threads(nt);
    return 0;
}