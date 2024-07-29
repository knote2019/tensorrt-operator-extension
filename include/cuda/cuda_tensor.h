#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#define __checkCudaTensorOp(cuda_mem_func)                                                                            \
    {                                                                                                                 \
        const cudaError __rt__ = cuda_mem_func;                                                                       \
        if (__rt__ != cudaSuccess) {                                                                                  \
            std::stringstream __ss__;                                                                                 \
            __ss__ << __FILE__ << ":" << __LINE__ << " >>> " << #cuda_mem_func << " == " << cudaGetErrorName(__rt__); \
            std::cout << __ss__.str() << std::endl;                                                                   \
            std::exit(EXIT_FAILURE);                                                                                  \
        }                                                                                                             \
    }

namespace cuda {

/*****************************************************************
 * Dims
 * **************************************************************/
class dims {
 public:
    static constexpr int MAX_DIMS{8};
    int nbDims;
    size_t d[MAX_DIMS];
};

namespace internal {
/*****************************************************************
 * memory
 * **************************************************************/
template <typename T>
class memory {
 private:
    size_t _size{};
    void* _gpu_data{};
    cudaDataType_t _gpu_data_type{};
    std::string _gpu_data_type_string{};
    std::string _gpu_memory_string{};

 public:
    size_t size() {
        return _size;
    }
    size_t bytes() {
        return _size * sizeof(T);
    }

 protected:
    explicit memory(size_t size) : _size(size) {
        // _gpu_data should be void* due to it will core dump when size = 0.
        __checkCudaTensorOp(cudaMalloc(&_gpu_data, _size * sizeof(T)));
        __checkCudaTensorOp(cudaMemset(_gpu_data, 0, _size * sizeof(T)));
        set_gpu_data_type();
        set_gpu_memory_name();
    }
    //---------------------------------
    void H2D(T* cpuData) {
        __checkCudaTensorOp(cudaMemcpy(_gpu_data, cpuData, _size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void D2H(T* cpuData) {
        __checkCudaTensorOp(cudaMemcpy(cpuData, _gpu_data, _size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    //---------------------------------
    std::string gpu_data_type_string() {
        return _gpu_data_type_string;
    }
    std::string gpu_memory_string() {
        return _gpu_memory_string;
    }
    T* gpu_data() {
        return static_cast<T*>(_gpu_data);
    }
    virtual cudaDataType_t gpu_data_type() {
        return _gpu_data_type;
    }
    //---------------------------------
    ~memory() noexcept {
        __checkCudaTensorOp(cudaFree(_gpu_data));
    }

 private:
    //---------------------------------
    void set_gpu_data_type() {
        if (std::is_same_v<T, int8_t>) {
            _gpu_data_type = CUDA_R_8I;
            _gpu_data_type_string = "int8_t";
        } else if (std::is_same_v<T, uint8_t>) {
            _gpu_data_type = CUDA_R_8U;
            _gpu_data_type_string = "uint8_t";
        } else if (std::is_same_v<T, int16_t>) {
            _gpu_data_type = CUDA_R_16I;
            _gpu_data_type_string = "int16_t";
        } else if (std::is_same_v<T, uint16_t>) {
            _gpu_data_type = CUDA_R_16U;
            _gpu_data_type_string = "uint16_t";
        } else if (std::is_same_v<T, int32_t>) {
            _gpu_data_type = CUDA_R_32I;
            _gpu_data_type_string = "int32_t";
        } else if (std::is_same_v<T, uint32_t>) {
            _gpu_data_type = CUDA_R_32U;
            _gpu_data_type_string = "uint32_t";
        } else if (std::is_same_v<T, float>) {
            _gpu_data_type = CUDA_R_32F;
            _gpu_data_type_string = "float";
        } else if (std::is_same_v<T, half>) {
            _gpu_data_type = CUDA_R_16F;
            _gpu_data_type_string = "half";
        } else if (std::is_same_v<T, nv_bfloat16>) {
            _gpu_data_type = CUDA_R_16BF;
            _gpu_data_type_string = "nv_bfloat16";
        }
    }
    void set_gpu_memory_name() {
        std::stringstream ss;
        ss << _gpu_data;
        _gpu_memory_string = ss.str();
    }
};
}  // namespace internal

/*****************************************************************
 * value
 * **************************************************************/
template <typename T>
class scalar : protected cuda::internal::memory<T> {
 private:
    std::string value_name{};
    T cpu_value{};
    bool is_gpu_func_invoked = false;
    std::string value_desc{};

 public:
    explicit scalar(const std::string& valueName, const T cpuValue)
        : cuda::internal::memory<T>(1), value_name(valueName), cpu_value(cpuValue) {
        if (valueName.empty()) {
            value_desc = "[" + this->gpu_memory_string() + "][" + this->gpu_data_type_string() + "]";
        } else {
            value_desc = "[" + value_name + "][" + this->gpu_data_type_string() + "]";
        }
        this->H2D(&cpu_value);
    }
    explicit scalar(const T cpuValue) : scalar("", cpuValue) {}
    T* GPU() {
        is_gpu_func_invoked = true;
        return this->gpu_data();
    }
    T CPU() {
        if (is_gpu_func_invoked) {
            this->D2H(&cpu_value);
            is_gpu_func_invoked = false;
        }
        return cpu_value;
    }
    void show() {
        auto old_flags = std::cout.flags();
        CPU();  // sync with GPU first.
        std::stringstream ss;
        ss << std::fixed << std::boolalpha;
        ss << std::endl << ">>> >>> >>>" << std::endl;
        ss << value_desc << " = " << std::to_string(cpu_value) << std::endl;
        ss << "<<< <<< <<<" << std::endl;
        std::cout << ss.str();
        std::cout.flags(old_flags);
    }
};

/*****************************************************************
 * Tensor
 * **************************************************************/
template <typename T>
class tensor : public cuda::internal::memory<T> {
 protected:
    std::string tensor_name;
    cuda::dims tensor_dims{};
    std::vector<T>* cpu_vector = nullptr;
    bool is_gpu_func_invoked = false;
    std::string tensor_desc;

 public:
    tensor(std::string tensor_name, size_t w) : tensor(std::move(tensor_name), {1, {w}}) {}
    tensor(std::string tensor_name, size_t h, size_t w) : tensor(std::move(tensor_name), {2, {h, w}}) {}
    tensor(std::string tensor_name, size_t c, size_t h, size_t w) : tensor(std::move(tensor_name), {3, {c, h, w}}) {}
    tensor(std::string tensor_name, size_t n, size_t c, size_t h, size_t w)
        : tensor(std::move(tensor_name), {4, {n, c, h, w}}) {}
    explicit tensor(size_t w) : tensor("", {1, {w}}) {}
    tensor(size_t h, size_t w) : tensor("", {2, {h, w}}) {}
    tensor(size_t c, size_t h, size_t w) : tensor("", {3, {c, h, w}}) {}
    tensor(size_t n, size_t c, size_t h, size_t w) : tensor("", {4, {n, c, h, w}}) {}
    tensor(std::string tensorName, cuda::dims tensorDims)
        : cuda::internal::memory<T>(calculate_dims_length(tensorDims)),
          tensor_name(std::move(tensorName)),
          tensor_dims(tensorDims) {
        if (tensor_name.empty()) {
            tensor_desc = "[" + this->gpu_memory_string() + "][" + dims_to_string(tensor_dims) + "][" +
                          this->gpu_data_type_string() + "]";
        } else {
            tensor_desc =
                "[" + tensor_name + "][" + dims_to_string(tensor_dims) + "][" + this->gpu_data_type_string() + "]";
        }
        cpu_vector = new std::vector<T>(calculate_dims_length(tensor_dims));
    }
    /*****************************************************************
     * init_with_random
     * **************************************************************/
    void init_with_random(unsigned long random_seed) {
        auto temp = std::vector<T>(1024);
        if constexpr (std::is_integral_v<T>) {
            for (size_t i = 0; i < temp.size(); i++) {
                init_value_with_random(&(temp.data()[i]), -9, 9, random_seed * (i + 1));
            }
        } else {
            for (size_t i = 0; i < temp.size(); i++) {
                init_value_with_random(&(temp.data()[i]), -2.0, 2.0, random_seed * (i + 1));
            }
        }
        init_with_data(temp.data(), temp.size());
    }
    /*****************************************************************
     * init_with_vector
     * **************************************************************/
    void init_with_vector(std::vector<T> vector) {
        init_with_data(vector.data(), vector.size());
    }
    /*****************************************************************
     * init_with_values
     * **************************************************************/
    void init_with_values(const std::initializer_list<T> values) {
        auto temp = std::vector<T>(values);
        init_with_data(temp.data(), temp.size());
    }
    /*****************************************************************
     * init_with_same
     * **************************************************************/
    void init_with_same(T value) {
        auto temp = std::vector<T>(1024);
        for (size_t i = 0; i < temp.size(); i++) {
            temp[i] = value;
        }
        init_with_data(temp.data(), temp.size());
    }
    /*****************************************************************
     * init_with_data
     * **************************************************************/
    void init_with_data(T* data, size_t size) {
        // calculate block number.
        size_t block_size = size;
        size_t block_num = cpu_vector->size() / size;
        size_t other_size = cpu_vector->size() - block_size * block_num;
        // copy blocks.
        for (int i = 0; i < block_num; i++) {
            std::copy(data, data + size, cpu_vector->begin() + i * block_size);
        }
        // copy others.
        std::copy(data, data + other_size, cpu_vector->begin() + block_size * block_num);
        this->H2D(cpu_vector->data());
    }

    T* GPU() {
        is_gpu_func_invoked = true;
        return this->gpu_data();
    }
    std::vector<T>* CPU() {
        if (is_gpu_func_invoked) {
            this->D2H(cpu_vector->data());
            is_gpu_func_invoked = false;
        }
        return cpu_vector;
    }
    virtual cudaDataType_t datatype() {
        return this->gpu_data_type();
    }
    void show() {
        auto old_flags = std::cout.flags();
        std::cout << std::endl << ">>> " << tensor_name << " = >>>" << std::endl;
        int max_line_size = 8;
        int max_line_count = 16;
        int lines_count = 0;
        for (size_t i = 0; i < cpu_vector->size(); i++) {
            std::cout << std::fixed << std::setw(12);
            std::cout << std::to_string(CPU()->data()[i]) << "\t";
            if ((i + 1) % max_line_size == 0) {  // next line.
                std::cout << std::endl;
                lines_count++;
                if (lines_count > max_line_count) {
                    std::cout << "... ... ..." << std::endl;
                    break;
                }
            }
        }
        std::cout << std::endl << "<<< <<< <<<" << std::endl;
        std::cout.flags(old_flags);
    }
    void compare_with_vector(std::vector<T>& vector) {
        compare_with_data(vector.data(), vector.size());
    }
    void compare_with_data(T* data, size_t size) {
        for (int i = 0; i < size; i++) {
            compare_value(*(data + i), CPU()->data()[i]);
        }
    }
    ~tensor() {
        if (cpu_vector) {
            delete cpu_vector;
            cpu_vector = nullptr;
        }
    }

 private:
    static size_t calculate_dims_length(cuda::dims dims) {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<>());
    }
    static std::string dims_to_string(const cuda::dims& d) {
        std::stringstream ss;
        for (int i = 0; i < d.nbDims; ++i) {
            ss << d.d[i];
            if (i < d.nbDims - 1) {
                ss << " x ";
            }
        }
        return ss.str();
    }
    static void init_value_with_random(T* value, T min, T max, int seed) {
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> distribution(min, max);
            std::mt19937 random_engine(seed);
            *value = distribution(random_engine);
        } else {
            std::uniform_real_distribution<float> distribution(min, max);
            std::mt19937 random_engine(seed);
            *value = distribution(random_engine);
        }
    }
    static void compare_value(T expectValue, T actualValue) {
        bool flag = false;
        if constexpr (std::is_integral_v<T>) {
            flag = (expectValue == actualValue);
        } else {
            float precision = 0.01;
            if (std::isnan(expectValue) || std::isnan(actualValue) || std::isinf(expectValue) ||
                std::isinf(actualValue)) {
                flag == (std::isnan(expectValue) && std::isnan(actualValue)) ||
                    (std::isinf(expectValue) && std::isinf(actualValue));
            } else {
                bool b1 = std::abs(expectValue - actualValue) / std::abs(expectValue) < precision;
                bool b2 = std::abs(expectValue - actualValue) / std::abs(actualValue) < precision;
                bool b3 = std::abs(expectValue - actualValue) < precision;
                flag = b1 || b2 || b3;
            }
        }
        if (!flag) {
            std::stringstream msg;
            msg << "compare_value failed:" << std::endl
                << "expect value: " << std::to_string(expectValue) << std::endl
                << "actual value: " << std::to_string(actualValue) << std::endl;
            throw std::runtime_error(msg.str());
        }
    }
};

}  // namespace cuda
