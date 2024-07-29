#include "common.h"


TEST(tensorrt, resnet50) {
    std::string model_path = "/stores/pytorch-models/tensorrt-operator-extension/sample_customLeakyReLU.onnx";
    std::string engine_path = "/tmp/sample_customLeakyReLU.engine";

    if (!TensorRTUtil::check_file_exist(engine_path)) {
        auto engine = TensorRTEngine(model_path);
        engine.config->setFlag(nvinfer1::BuilderFlag::kFP16);
        auto input = engine.network->getInput(0);
        if (input->getDimensions().d[0] == -1) {
            // set for dynamic batch size.
            nvinfer1::IOptimizationProfile* profile = engine.builder->createOptimizationProfile();
            int c = 1;
            int h = 5;
            int w = 5;
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

    auto input_dims = context->getTensorShape("input");
    auto output_dims = context->getTensorShape("output");
    TensorRTUtil::print_dims("input_dims", input_dims);
    TensorRTUtil::print_dims("output_dims", output_dims);

    size_t input_length = TensorRTUtil::calculate_dims_length(input_dims);
    size_t output_length = TensorRTUtil::calculate_dims_length(output_dims);
    std::cout << "input_length = " << input_length << std::endl;
    std::cout << "output_length = " << output_length << std::endl;

    auto input_gpu = cuda::vector<float>(input_length);
    auto output_gpu = cuda::vector<float>(output_length);

    void* bindings[] = {input_gpu.GPU(), output_gpu.GPU()};

    // -----------------------------------------------------------------------------------------------------------------
    // input.
    std::string image_path = TensorRTUtil::root_path() + "src/image/flamingo/ILSVRC2012_val_00000356.JPEG";
    std::cout << image_path << std::endl;

    cv::Size shape{224, 224};
    cv::Scalar mean{0.485, 0.456, 0.406};
    cv::Scalar std{0.229, 0.224, 0.225};

    cv::Mat image = cv::imread(image_path);
    cv::Mat input = TensorRTUtil::pre_process(image, shape, mean, std);
    auto input_cpu = std::vector<float>(input_length);
    input_cpu.assign(input.begin<float>(), input.end<float>());
    input_gpu.set(&input_cpu);

    // run.
    context->executeV2(bindings);

    // output.
    auto output_cpu = std::vector<float>(output_length);
    output_gpu.get(&output_cpu);

    unsigned int class_id = TensorRTUtil::get_max_value_index(&output_cpu);
    std::string class_name = TensorRTUtil::get_imagenet_class_name(class_id);

    std::cout << "class_id = " << class_id << std::endl;
    std::cout << "class_name = " << class_name << std::endl;

    TensorRTUtil::put_text(image, class_name);

    // show image.
    cv::imshow("image", image);
    cv::waitKey(0);
    std::cout << std::endl;
}
