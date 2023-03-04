#include "../memory_op.h"

#include <complex>

#include <cuda_runtime.h>
#include <thrust/complex.h>

#define THREADS_PER_BLOCK 256

namespace container {
namespace op {

template <typename FPTYPE_out, typename FPTYPE_in>
__global__ void cast_memory(
        FPTYPE_out* out,
        const FPTYPE_in* in,
        const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    out[idx] = static_cast<FPTYPE_out>(in[idx]);
}

template <typename FPTYPE_out, typename FPTYPE_in>
__global__ void cast_memory(
        std::complex<FPTYPE_out>* out,
        const std::complex<FPTYPE_in>* in,
        const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    auto* _out = reinterpret_cast<thrust::complex<FPTYPE_out>*>(out);
    const auto* _in = reinterpret_cast<const thrust::complex<FPTYPE_in>*>(in);
    _out[idx] = static_cast<thrust::complex<FPTYPE_out>>(_in[idx]);
}

template <typename FPTYPE>
void resize_memory_op<FPTYPE, container::DEVICE_GPU>::operator()(
    const container::DEVICE_GPU* dev,
    FPTYPE*& arr, 
    const size_t size,
    const char* record_in)
{
  if (arr != nullptr) {
    delete_memory_op<FPTYPE, container::DEVICE_GPU>()(dev, arr);
  }
  cudaMalloc((void **)&arr, sizeof(FPTYPE) * size);
}

template <typename FPTYPE>
void set_memory_op<FPTYPE, container::DEVICE_GPU>::operator()(
    const container::DEVICE_GPU* dev,
    FPTYPE* arr, 
    const int var, 
    const size_t size) 
{
  cudaMemset(arr, var, sizeof(FPTYPE) * size);  
}

template <typename FPTYPE> 
void synchronize_memory_op<FPTYPE, container::DEVICE_CPU, container::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size) 
{
  cudaMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost);  
}

template <typename FPTYPE> 
void synchronize_memory_op<FPTYPE, container::DEVICE_GPU, container::DEVICE_CPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size) 
{
  cudaMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice);  
}

template <typename FPTYPE> 
void synchronize_memory_op<FPTYPE, container::DEVICE_GPU, container::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size) 
{
  cudaMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, cudaMemcpyDeviceToDevice);  
}

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_GPU, container::DEVICE_GPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {
        const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cast_memory<<<block, THREADS_PER_BLOCK>>>(arr_out, arr_in, size);
    }
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_GPU, container::DEVICE_CPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {
        FPTYPE_in * arr = nullptr;
        cudaMalloc((void **)&arr, sizeof(FPTYPE_in) * size);
        cudaMemcpy(arr, arr_in, sizeof(FPTYPE_in) * size, cudaMemcpyHostToDevice);
        const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cast_memory<<<block, THREADS_PER_BLOCK>>>(arr_out, arr, size);
        cudaFree(arr);
    }
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_CPU, container::DEVICE_GPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {
        auto * arr = (FPTYPE_in*) malloc(sizeof(FPTYPE_in) * size);
        cudaMemcpy(arr, arr_in, sizeof(FPTYPE_in) * size, cudaMemcpyDeviceToHost);
        for (int ii = 0; ii < size; ii++) {
            arr_out[ii] = static_cast<FPTYPE_out>(arr[ii]);
        }
        free(arr);
    }
};

template <typename FPTYPE>
void delete_memory_op<FPTYPE, container::DEVICE_GPU>::operator() (
    const container::DEVICE_GPU* dev,
    FPTYPE* arr) 
{
  cudaFree(arr);
}

template struct resize_memory_op<int, container::DEVICE_GPU>;
template struct resize_memory_op<int64_t, container::DEVICE_GPU>;
template struct resize_memory_op<float, container::DEVICE_GPU>;
template struct resize_memory_op<double, container::DEVICE_GPU>;
template struct resize_memory_op<std::complex<float>, container::DEVICE_GPU>;
template struct resize_memory_op<std::complex<double>, container::DEVICE_GPU>;

template struct set_memory_op<int, container::DEVICE_GPU>;
template struct set_memory_op<int64_t , container::DEVICE_GPU>;
template struct set_memory_op<float, container::DEVICE_GPU>;
template struct set_memory_op<double, container::DEVICE_GPU>;
template struct set_memory_op<std::complex<float>, container::DEVICE_GPU>;
template struct set_memory_op<std::complex<double>, container::DEVICE_GPU>;

template struct synchronize_memory_op<int, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<int, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<int, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<int64_t, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<int64_t, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<int64_t, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<float, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<float, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<float, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<double, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<double, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<double, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<std::complex<float>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<std::complex<double>, container::DEVICE_GPU, container::DEVICE_GPU>;

template struct cast_memory_op<float, float, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<double, double, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<float, double, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<double, float, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory_op<float, float, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<double, double, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<float, double, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<double, float, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<float>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<double>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory_op<float, float, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<double, double, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<float, double, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<double, float, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;

template struct delete_memory_op<int, container::DEVICE_GPU>;
template struct delete_memory_op<int64_t, container::DEVICE_GPU>;
template struct delete_memory_op<float, container::DEVICE_GPU>;
template struct delete_memory_op<double, container::DEVICE_GPU>;
template struct delete_memory_op<std::complex<float>, container::DEVICE_GPU>;
template struct delete_memory_op<std::complex<double>, container::DEVICE_GPU>;

} // end of namespace container
} // end of namespace op