#pragma once

#include <cuda_runtime.h>

#define checkCUDA(cuda_func)                                                                                      \
    {                                                                                                             \
        const cudaError __rt__ = cuda_func;                                                                       \
        if (__rt__ != cudaSuccess) {                                                                              \
            std::stringstream __ss__;                                                                             \
            __ss__ << __FILE__ << ":" << __LINE__ << " >>> " << #cuda_func << " == " << cudaGetErrorName(__rt__); \
            std::cout << __ss__.str() << std::endl;                                                               \
            std::exit(EXIT_FAILURE);                                                                              \
        }                                                                                                         \
    }

namespace cuda {

template <typename T>
class value {
 private:
    void* _pGpuData{};

 public:
    explicit value(T value) {
        checkCUDA(cudaMalloc(&_pGpuData, sizeof(T)));
        checkCUDA(cudaMemcpy(_pGpuData, &value, sizeof(T), cudaMemcpyHostToDevice));
        checkCUDA(cudaMemset(_pGpuData, 0, sizeof(T)));
    }
    T* GPU() {
        return static_cast<T*>(_pGpuData);
    }
    T CPU() {
        T value;
        checkCUDA(cudaMemcpy(&value, _pGpuData, sizeof(T), cudaMemcpyDeviceToHost));
        return value;
    }
    ~value() noexcept {
        checkCUDA(cudaFree(_pGpuData));
    }
};

/*****************************************************************
 * vector
 * **************************************************************/
template <typename T>
class vector {
 private:
    void* _pGpuData{};
    size_t _size{};

 public:
    explicit vector(size_t size) : _size(size) {
        // _pGpuData should be void* due to it will core dump when size = 0.
        checkCUDA(cudaMalloc(&_pGpuData, size * sizeof(T)));
        checkCUDA(cudaMemset(_pGpuData, 0, size * sizeof(T)));
    }
    size_t size() {
        return _size;
    }
    size_t bytes() {
        return _size * sizeof(T);
    }
    T* GPU() {
        return static_cast<T*>(_pGpuData);
    }
    // SET.
    void set(std::vector<T>* pVector) {
        this->set(pVector->data(), pVector->size());
    }
    void set(T* pData, size_t size) {
        int copy_size = std::min(_size, size);
        checkCUDA(cudaMemcpy(_pGpuData, pData, copy_size * sizeof(T), cudaMemcpyHostToDevice));
    }
    // GET.
    void get(std::vector<T>* pVector) {
        this->get(pVector->data(), pVector->size());
    }
    void get(T* pData, size_t size) {
        int copy_size = std::min(_size, size);
        checkCUDA(cudaMemcpy(pData, _pGpuData, copy_size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    ~vector() noexcept {
        checkCUDA(cudaFree(_pGpuData));
    }
};

/*****************************************************************
 * stream
 * **************************************************************/
class stream {
 private:
    cudaStream_t _cuda_stream{};

 public:
    explicit stream() {
        checkCUDA(cudaStreamCreate(&_cuda_stream));
    }
    cudaStream_t& operator()() {
        return _cuda_stream;
    }
    ~stream() noexcept {
        if (_cuda_stream) {
            checkCUDA(cudaStreamDestroy(_cuda_stream));
        }
    }
};

/*****************************************************************
 * graph
 * **************************************************************/
class graph {
 private:
    cudaGraph_t _cuda_graph{};

 public:
    explicit graph(unsigned int flags) {
        checkCUDA(cudaGraphCreate(&_cuda_graph, flags));
    }
    cudaGraph_t& operator()() {
        return _cuda_graph;
    }
    ~graph() noexcept {
        if (_cuda_graph) {
            checkCUDA(cudaGraphDestroy(_cuda_graph));
        }
    }
};

/*****************************************************************
 * event
 * **************************************************************/
class event {
 private:
    cudaEvent_t _cuda_event{};

 public:
    explicit event() {
        checkCUDA(cudaEventCreate(&_cuda_event));
    }
    cudaEvent_t& operator()() {
        return _cuda_event;
    }
    ~event() noexcept {
        if (_cuda_event) {
            checkCUDA(cudaEventDestroy(_cuda_event));
        }
    }
};

}  // namespace cuda
