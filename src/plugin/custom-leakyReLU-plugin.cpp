#include "custom-leakyReLU-plugin.h"

#include <cstring>
#include <iostream>
#include <map>
#include <utility>

/******************************************************************/
/******************** CustomLeakyReLU的核函数接口部分 ****************/
/******************************************************************/
void customLeakyReLUImpl(const float* inputs, float* outputs, float alpha, int nElements, cudaStream_t stream);

namespace custom {

REGISTER_TENSORRT_PLUGIN(CustomLeakyReLUPluginCreator);

nvinfer1::PluginFieldCollection CustomLeakyReLUPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CustomLeakyReLUPluginCreator::mAttrs{};

/******************************************************************/
/*********************CustomLeakyReLUPlugin部分*********************/
/******************************************************************/

CustomLeakyReLUPlugin::CustomLeakyReLUPlugin(std::string name, float alpha) : mName(std::move(name)) {
    mParams.alpha = alpha;
    if (alpha < 0.0F) {
        std::cout << "alpha can't less than .0.0f" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

CustomLeakyReLUPlugin::CustomLeakyReLUPlugin(std::string name, const void* buffer, size_t length)
    : mName(std::move(name)) {
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomLeakyReLUPlugin::~CustomLeakyReLUPlugin() = default;

const char* CustomLeakyReLUPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char* CustomLeakyReLUPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

int32_t CustomLeakyReLUPlugin::getNbOutputs() const noexcept {
    return 1;
}

size_t CustomLeakyReLUPlugin::getSerializationSize() const noexcept {
    return sizeof(mParams);
}

const char* CustomLeakyReLUPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

nvinfer1::DataType CustomLeakyReLUPlugin::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                                            int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

nvinfer1::DimsExprs CustomLeakyReLUPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs,
                                                               int32_t nbInputs,
                                                               nvinfer1::IExprBuilder& exprBuilder) noexcept {
    return inputs[0];
}

size_t CustomLeakyReLUPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
                                               const nvinfer1::PluginTensorDesc* outputs,
                                               int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t CustomLeakyReLUPlugin::initialize() noexcept {
    return 0;
}

void CustomLeakyReLUPlugin::terminate() noexcept {}

void CustomLeakyReLUPlugin::serialize(void* buffer) const noexcept {
    memcpy(buffer, &mParams, sizeof(mParams));
}

void CustomLeakyReLUPlugin::destroy() noexcept {
    delete this;
}

int32_t CustomLeakyReLUPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                       const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                       void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++) {
        nElements *= inputDesc[0].dims.d[i];
    }

    customLeakyReLUImpl(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), mParams.alpha, nElements,
                        stream);

    return 0;
}

nvinfer1::IPluginV2DynamicExt* CustomLeakyReLUPlugin::clone() const noexcept {
    try {
        auto p = new CustomLeakyReLUPlugin(mName, &mParams, sizeof(mParams));
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (std::exception const& e) {
        std::cout << "clone failed due to " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

bool CustomLeakyReLUPlugin::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut,
                                                      int32_t nbInputs, int32_t nbOutputs) noexcept {
    if (pos < nbInputs) {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
    if (pos < nbInputs + nbOutputs) {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

void CustomLeakyReLUPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
                                            const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}
void CustomLeakyReLUPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}
void CustomLeakyReLUPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas,
                                            nvinfer1::IGpuAllocator* gpuAllocator) noexcept {}
void CustomLeakyReLUPlugin::detachFromContext() noexcept {}

/******************************************************************/
/*********************CustomLeakyReLUPluginCreator部分********************/
/******************************************************************/

CustomLeakyReLUPluginCreator::CustomLeakyReLUPluginCreator() {
    mAttrs.emplace_back("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mFC.nbFields = mAttrs.size();
    mFC.fields = mAttrs.data();
}

CustomLeakyReLUPluginCreator::~CustomLeakyReLUPluginCreator() = default;

const char* CustomLeakyReLUPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char* CustomLeakyReLUPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const char* CustomLeakyReLUPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

nvinfer1::IPluginV2DynamicExt* CustomLeakyReLUPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
    try {
        float alpha = 0.0;
        for (int i = 0; i < fc->nbFields; ++i) {
            if (std::string(fc->fields[i].name) == "alpha" &&
                fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32) {
                alpha = *(static_cast<const float*>(fc->fields[i].data));
            }
        }
        return new CustomLeakyReLUPlugin(name, alpha);
    } catch (std::exception const& e) {
        std::cout << " create plugin failed due to " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

nvinfer1::IPluginV2DynamicExt* CustomLeakyReLUPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                               size_t serialLength) noexcept {
    try {
        return new CustomLeakyReLUPlugin(name, serialData, serialLength);
    } catch (std::exception const& e) {
        std::cout << " deserialize plugin failed due to " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void CustomLeakyReLUPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const nvinfer1::PluginFieldCollection* CustomLeakyReLUPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

}  // namespace custom
