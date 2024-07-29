#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <iostream>
#include <string>
#include <utility>

#include "tensorrt_logger.h"
#include "tensorrt_util.h"

class TensorRTEngine {
 private:
    std::string _onnx_or_engine;
    TensorRTLogger _logger;

 public:
    nvinfer1::IBuilder* builder{};
    nvinfer1::INetworkDefinition* network{};
    nvonnxparser::IParser* parser{};
    nvinfer1::IBuilderConfig* config{};
    nvinfer1::IHostMemory* memory{};
    nvinfer1::IRuntime* runtime{};
    nvinfer1::ICudaEngine* engine{};
    std::vector<nvinfer1::IExecutionContext*> contexts{};

 public:
    explicit TensorRTEngine(std::string onnx_or_engine) : _onnx_or_engine(std::move(onnx_or_engine)) {
        if (_onnx_or_engine.find(".onnx") != std::string::npos) {
            builder = nvinfer1::createInferBuilder(_logger);
            network = builder->createNetworkV2(1);
            parser = nvonnxparser::createParser(*network, _logger);
            parser->parseFromFile(_onnx_or_engine.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
            config = builder->createBuilderConfig();
        }
    };

    void setup() {
        std::cout << "setup " << _onnx_or_engine << "'s engine start >>>" << std::endl;
        runtime = nvinfer1::createInferRuntime(_logger);
        if (_onnx_or_engine.find(".onnx") != std::string::npos) {
            memory = builder->buildSerializedNetwork(*network, *config);
            engine = runtime->deserializeCudaEngine(memory->data(), memory->size());
        } else {
            auto engine_vector = std::vector<char>();
            TensorRTUtil::read_file(_onnx_or_engine, &engine_vector);
            engine = runtime->deserializeCudaEngine(engine_vector.data(), engine_vector.size());
        }
        std::cout << "setup " << _onnx_or_engine << "'s engine stop <<<" << std::endl;
    }

    void save(const std::string& model_engine_file) const {
        std::cout << "save " << _onnx_or_engine << "'s engine start >>>" << std::endl;
        if (_onnx_or_engine.find(".onnx") != std::string::npos) {
            TensorRTUtil::write_file(model_engine_file, reinterpret_cast<char*>(engine->serialize()->data()),
                                     engine->serialize()->size());
        }
        std::cout << "save " << _onnx_or_engine << "'s engine stop <<<" << std::endl;
    }

    nvinfer1::IExecutionContext* create_context() {
        auto context = engine->createExecutionContext();
        contexts.emplace_back(context);
        return context;
    }

    ~TensorRTEngine() noexcept {
        for (auto context : contexts) {
            delete_ptr(context);
        }
        delete_ptr(engine);
        delete_ptr(runtime);
        delete_ptr(memory);
        delete_ptr(config);
        delete_ptr(parser);
        delete_ptr(network);
        delete_ptr(builder);
    }

 private:
    template <typename T>
    static void delete_ptr(T*& p) {
        if (p) {
            delete p;
            p = nullptr;
        }
    }
};
