#pragma once

#include <NvInfer.h>

#include <string>
#include <vector>

namespace custom {

static const char* PLUGIN_NAME{"customLeakyReLU"};
static const char* PLUGIN_VERSION{"1"};

class CustomLeakyReLUPlugin : public nvinfer1::IPluginV2DynamicExt {
 private:
    const std::string mName;
    std::string mNamespace;
    struct {
        float alpha;
    } mParams;

 public:
    CustomLeakyReLUPlugin() = delete;
    CustomLeakyReLUPlugin(std::string name, float alpha);
    CustomLeakyReLUPlugin(std::string name, const void* buffer, size_t length);
    ~CustomLeakyReLUPlugin() override;

    [[nodiscard]] const char* getPluginType() const noexcept override;
    [[nodiscard]] const char* getPluginVersion() const noexcept override;
    [[nodiscard]] int32_t getNbOutputs() const noexcept override;
    [[nodiscard]] size_t getSerializationSize() const noexcept override;
    [[nodiscard]] const char* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* input, int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

    [[nodiscard]] IPluginV2DynamicExt* clone() const noexcept override;

    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOuts, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    void attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas,
                         nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;
};

class CustomLeakyReLUPluginCreator : public nvinfer1::IPluginCreator {
 private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mAttrs;
    std::string mNamespace;

 public:
    CustomLeakyReLUPluginCreator();
    ~CustomLeakyReLUPluginCreator() override;

    [[nodiscard]] const char* getPluginName() const noexcept override;
    [[nodiscard]] const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    [[nodiscard]] const char* getPluginNamespace() const noexcept override;
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,
                                                const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
                                                     size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
};

}  // namespace custom
