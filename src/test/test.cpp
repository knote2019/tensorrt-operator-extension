#include "common.h"

TEST(tensorrt, operator_extension) {
    std::string model_path = "/stores/pytorch-models/tensorrt-operator-extension/model_with_customLeakyReLU.onnx";
    std::string engine_path = "/tmp/model_with_customLeakyReLU.engine";

    if (!TensorRTUtil::check_file_exist(engine_path)) {
        auto engine = TensorRTEngine(model_path);
        engine.config->setFlag(nvinfer1::BuilderFlag::kFP16);
        auto input = engine.network->getInput(0);
        if (input->getDimensions().d[0] == -1) {
            // set for dynamic batch size.
            nvinfer1::IOptimizationProfile* profile = engine.builder->createOptimizationProfile();
            int c = 1, h = 5, w = 5;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims{4, {1, c, h, w}});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims{4, {5, c, h, w}});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims{4, {10, c, h, w}});
            engine.config->addOptimizationProfile(profile);
        }
        engine.setup();
        engine.save(engine_path);
    }

    // -----------------------------------------------------------------------------------------------------------------
    auto engine = TensorRTEngine(engine_path);
    engine.setup();
    auto context = engine.create_context();

    // set input dims.
    context->setInputShape("input", nvinfer1::Dims{4, {1, 1, 5, 5}});
    TensorRTUtil::print_dims("input_dims", context->getTensorShape("input"));
    TensorRTUtil::print_dims("output_dims", context->getTensorShape("output"));

    auto input = cuda::tensor<float>(1, 1, 5, 5);
    auto output = cuda::tensor<float>(1, 3, 5, 5);

    // set input.
    input.init_with_values({0.7576, 0.2793, 0.4031, 0.7347, 0.0293, 0.7999, 0.3971, 0.7544, 0.5695,
                            0.4388, 0.6387, 0.5247, 0.6826, 0.3051, 0.4635, 0.4550, 0.5725, 0.4980,
                            0.9371, 0.6556, 0.3138, 0.1980, 0.4162, 0.2843, 0.3398});
    input.show();

    // run.
    auto bindings = std::vector<void*>();
    bindings.emplace_back(input.GPU());
    bindings.emplace_back(output.GPU());
    context->executeV2(bindings.data());

    // check output.
    output.show();
}
