#include <iostream>
#include <complex>
#include <string.h>
#include "memory_op.h"

namespace container {
namespace op {

template <typename FPTYPE> 
struct resize_memory_op<FPTYPE, container::DEVICE_CPU> {
  void operator()(const container::DEVICE_CPU* dev, FPTYPE*& arr, const size_t size, const char* /*record_in*/) {
    if (arr != nullptr) {
      free(arr);
    }
    arr = (FPTYPE*) malloc(sizeof(FPTYPE) * size);
  }
};

template <typename FPTYPE> 
struct set_memory_op<FPTYPE, container::DEVICE_CPU> {
  void operator()(const container::DEVICE_CPU* dev, FPTYPE* arr, const int var, const size_t size) {
    memset(arr, var, sizeof(FPTYPE) * size);
  }
};

template <typename FPTYPE> 
struct synchronize_memory_op<FPTYPE, container::DEVICE_CPU, container::DEVICE_CPU> {
  void operator()(FPTYPE* arr_out,
                  const FPTYPE* arr_in, 
                  const size_t size) {
    memcpy(arr_out, arr_in, sizeof(FPTYPE) * size);
  }
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_CPU, container::DEVICE_CPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {
        for (int ii = 0; ii < size; ii++) {
            arr_out[ii] = static_cast<FPTYPE_out>(arr_in[ii]);
        }
    }
};

template <typename FPTYPE>
struct delete_memory_op<FPTYPE, container::DEVICE_CPU> {
  void operator()(const container::DEVICE_CPU* dev, FPTYPE* arr) {
    free(arr);
  }
};

template struct resize_memory_op<int, container::DEVICE_CPU>;
template struct resize_memory_op<int64_t, container::DEVICE_CPU>;
template struct resize_memory_op<float, container::DEVICE_CPU>;
template struct resize_memory_op<double, container::DEVICE_CPU>;
template struct resize_memory_op<std::complex<float>, container::DEVICE_CPU>;
template struct resize_memory_op<std::complex<double>, container::DEVICE_CPU>;

template struct set_memory_op<int, container::DEVICE_CPU>;
template struct set_memory_op<int64_t, container::DEVICE_CPU>;
template struct set_memory_op<float, container::DEVICE_CPU>;
template struct set_memory_op<double, container::DEVICE_CPU>;
template struct set_memory_op<std::complex<float>, container::DEVICE_CPU>;
template struct set_memory_op<std::complex<double>, container::DEVICE_CPU>;

template struct synchronize_memory_op<int, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<int64_t, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<float, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<double, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<std::complex<float>, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct synchronize_memory_op<std::complex<double>, container::DEVICE_CPU, container::DEVICE_CPU>;

template struct cast_memory_op<float, float, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<double, double, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<float, double, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<double, float, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<float>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<double>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_CPU>;
template struct cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_CPU>;

template struct delete_memory_op<int, container::DEVICE_CPU>;
template struct delete_memory_op<int64_t, container::DEVICE_CPU>;
template struct delete_memory_op<float, container::DEVICE_CPU>;
template struct delete_memory_op<double, container::DEVICE_CPU>;
template struct delete_memory_op<std::complex<float>, container::DEVICE_CPU>;
template struct delete_memory_op<std::complex<double>, container::DEVICE_CPU>;

#if !(defined(__CUDA) || defined(__ROCM))
template <typename FPTYPE>
struct resize_memory_op<FPTYPE, container::DEVICE_GPU> {
    void operator()(const container::DEVICE_GPU* dev, FPTYPE*& arr, const size_t size, const char* record_in = nullptr) {}
};

template <typename FPTYPE>
struct set_memory_op<FPTYPE, container::DEVICE_GPU> {
    void operator()(const container::DEVICE_GPU* dev, FPTYPE* arr, const int var, const size_t size) {}
};

template <typename FPTYPE>
struct synchronize_memory_op<FPTYPE, container::DEVICE_GPU, container::DEVICE_GPU> {
    void operator()(FPTYPE* arr_out,
                    const FPTYPE* arr_in,
                    const size_t size) {}
};

template <typename FPTYPE>
struct synchronize_memory_op<FPTYPE, container::DEVICE_GPU, container::DEVICE_CPU> {
    void operator()(FPTYPE* arr_out,
                    const FPTYPE* arr_in,
                    const size_t size) {}
};

template <typename FPTYPE>
struct synchronize_memory_op<FPTYPE, container::DEVICE_CPU, container::DEVICE_GPU> {
    void operator()(FPTYPE* arr_out,
                    const FPTYPE* arr_in,
                    const size_t size) {}
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_GPU, container::DEVICE_GPU> {
    void operator()(const container::DEVICE_GPU* dev_out,
                    const container::DEVICE_GPU* dev_in,
                    FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {}
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_GPU, container::DEVICE_CPU> {
    void operator()(const container::DEVICE_GPU* dev_out,
                    const container::DEVICE_CPU* dev_in,
                    FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {}
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, container::DEVICE_CPU, container::DEVICE_GPU> {
    void operator()(const container::DEVICE_CPU* dev_out,
                    const container::DEVICE_GPU* dev_in,
                    FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {}
};

template <typename FPTYPE>
struct delete_memory_op<FPTYPE, container::DEVICE_GPU> {
    void operator()(const container::DEVICE_GPU* dev, FPTYPE* arr) {}
};

template struct resize_memory_op<int, container::DEVICE_GPU>;
template struct resize_memory_op<int64_t, container::DEVICE_GPU>;
template struct resize_memory_op<float, container::DEVICE_GPU>;
template struct resize_memory_op<double, container::DEVICE_GPU>;
template struct resize_memory_op<std::complex<float>, container::DEVICE_GPU>;
template struct resize_memory_op<std::complex<double>, container::DEVICE_GPU>;

template struct set_memory_op<int, container::DEVICE_GPU>;
template struct set_memory_op<int64_t, container::DEVICE_GPU>;
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
#endif

}
}