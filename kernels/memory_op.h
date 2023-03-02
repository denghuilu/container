#ifndef CONTAINER_MEMORY_H_
#define CONTAINER_MEMORY_H_

#include <vector>
#include <complex>
#include <stddef.h>

#include "../tensor_types.h"

namespace container {
namespace op {

/**
 * @brief A functor to resize memory allocation.
 * @tparam FPTYPE Floating-point type of the allocated memory.
 * @tparam Device Device type where the memory will be allocated.
 */
template <typename FPTYPE, typename Device>
struct resize_memory_op {
    /**
     * @brief Resize memory allocation.
     *
     * @param dev Device where the memory will be allocated.
     * @param arr Pointer to the allocated memory.
     * @param size New size of the allocated memory.
     * @param record_in Optional message to record the resize operation.
     */
    void operator()(const Device* dev, FPTYPE*& arr, const size_t size, const char* record_in = nullptr);
};

/**
 * @brief A functor to set memory to a constant value.
 * @tparam FPTYPE Floating-point type of the memory.
 * @tparam Device Device type where the memory is allocated.
 */
template <typename FPTYPE, typename Device>
struct set_memory_op {
    /**
     * @brief Set memory to a constant value.
     *
     * @param dev Device where the memory is allocated.
     * @param arr Pointer to the memory.
     * @param var Constant value to set.
     * @param size Size of the memory to set.
     */
    void operator()(const Device* dev, FPTYPE* arr, const int var, const size_t size);
};

/**
 * @brief Synchronizes memory between devices.
 *
 * This class synchronizes memory between two different devices.
 *
 * @tparam FPTYPE The type of data in the arrays.
 * @tparam Device_out The output device.
 * @tparam Device_in The input device.
 */
template <typename FPTYPE, typename Device_out, typename Device_in>
struct synchronize_memory_op {
    /**
     * @brief Synchronizes memory between devices.
     *
     * This method synchronizes memory between two different devices.
     *
     * @param dev_out The output device.
     * @param dev_in The input device.
     * @param arr_out The output array.
     * @param arr_in The input array.
     * @param size The size of the array.
     */
    void operator()(
        FPTYPE* arr_out,
        const FPTYPE* arr_in,
        const size_t size);
};

/**
 * @brief Casts memory between devices.
 *
 * This class casts memory between two different devices.
 *
 * @tparam FPTYPE_out The output data type.
 * @tparam FPTYPE_in The input data type.
 * @tparam Device_out The output device.
 * @tparam Device_in The input device.
 */
template <typename FPTYPE_out, typename FPTYPE_in, typename Device_out, typename Device_in>
struct cast_memory_op {
    /**
     * @brief Casts memory between devices.
     *
     * This method casts memory between two different devices.
     *
     * @param dev_out The output device.
     * @param dev_in The input device.
     * @param arr_out The output array.
     * @param arr_in The input array.
     * @param size The size of the array.
     */
    void operator()(
        const Device_out* dev_out,
        const Device_in* dev_in,
        FPTYPE_out* arr_out,
        const FPTYPE_in* arr_in,
        const size_t size);
};


/**
 * @brief Deletes memory on a device.
 *
 * This class deletes memory on a device.
 *
 * @tparam FPTYPE The type of data in the array.
 * @tparam Device The device.
 */
template <typename FPTYPE, typename Device>
struct delete_memory_op {
    /**
     * @brief Deletes memory on a device.
     *
     * This method deletes memory on a device.
     *
     * @param dev The device.
     * @param arr The array to be deleted.
     */
    void operator()(const Device* dev, FPTYPE* arr);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
// Partially specialize operator for container::GpuDevice.
template <typename FPTYPE>
struct resize_memory_op<FPTYPE, container::DEVICE_GPU> {
void operator()(const container::DEVICE_GPU* dev, FPTYPE*& arr, const size_t size, const char* record_in = nullptr);
};

template <typename FPTYPE>
struct set_memory_op<FPTYPE, container::DEVICE_GPU> {
void operator()(const container::DEVICE_GPU* dev, FPTYPE* arr, const int var, const size_t size);
};

template <typename FPTYPE>
struct synchronize_memory_op<FPTYPE, container::DEVICE_CPU, container::DEVICE_GPU> {
void operator()(
  FPTYPE* arr_out,
  const FPTYPE* arr_in,
  const size_t size);
};

template <typename FPTYPE>
struct synchronize_memory_op<FPTYPE, container::DEVICE_GPU, container::DEVICE_CPU> {
void operator()(
  FPTYPE* arr_out,
  const FPTYPE* arr_in,
  const size_t size);
};

template <typename FPTYPE>
struct synchronize_memory_op<FPTYPE, container::DEVICE_GPU, container::DEVICE_GPU> {
void operator()(
  FPTYPE* arr_out,
  const FPTYPE* arr_in,
  const size_t size);
};

template <typename FPTYPE>
struct delete_memory_op<FPTYPE, container::DEVICE_GPU> {
void operator()(const container::DEVICE_GPU* dev, FPTYPE* arr);
};
#endif
// __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

} // namespace op
} // namespace container

using resmem_sh_op = container::op::resize_memory_op<float, container::DEVICE_CPU>;
using resmem_dh_op = container::op::resize_memory_op<double, container::DEVICE_CPU>;
using resmem_ch_op = container::op::resize_memory_op<std::complex<float>, container::DEVICE_CPU>;
using resmem_zh_op = container::op::resize_memory_op<std::complex<double>, container::DEVICE_CPU>;

using resmem_sd_op = container::op::resize_memory_op<float, container::DEVICE_GPU>;
using resmem_dd_op = container::op::resize_memory_op<double, container::DEVICE_GPU>;
using resmem_cd_op = container::op::resize_memory_op<std::complex<float>, container::DEVICE_GPU>;
using resmem_zd_op = container::op::resize_memory_op<std::complex<double>, container::DEVICE_GPU>;

using setmem_sh_op = container::op::set_memory_op<float, container::DEVICE_CPU>;
using setmem_dh_op = container::op::set_memory_op<double, container::DEVICE_CPU>;
using setmem_ch_op = container::op::set_memory_op<std::complex<float>, container::DEVICE_CPU>;
using setmem_zh_op = container::op::set_memory_op<std::complex<double>, container::DEVICE_CPU>;

using setmem_sd_op = container::op::set_memory_op<float, container::DEVICE_GPU>;
using setmem_dd_op = container::op::set_memory_op<double, container::DEVICE_GPU>;
using setmem_cd_op = container::op::set_memory_op<std::complex<float>, container::DEVICE_GPU>;
using setmem_zd_op = container::op::set_memory_op<std::complex<double>, container::DEVICE_GPU>;

using delmem_sh_op = container::op::delete_memory_op<float, container::DEVICE_CPU>;
using delmem_dh_op = container::op::delete_memory_op<double, container::DEVICE_CPU>;
using delmem_ch_op = container::op::delete_memory_op<std::complex<float>, container::DEVICE_CPU>;
using delmem_zh_op = container::op::delete_memory_op<std::complex<double>, container::DEVICE_CPU>;

using delmem_sd_op = container::op::delete_memory_op<float, container::DEVICE_GPU>;
using delmem_dd_op = container::op::delete_memory_op<double, container::DEVICE_GPU>;
using delmem_cd_op = container::op::delete_memory_op<std::complex<float>, container::DEVICE_GPU>;
using delmem_zd_op = container::op::delete_memory_op<std::complex<double>, container::DEVICE_GPU>;

using syncmem_s2s_h2h_op = container::op::synchronize_memory_op<float, container::DEVICE_CPU, container::DEVICE_CPU>;
using syncmem_s2s_h2d_op = container::op::synchronize_memory_op<float, container::DEVICE_GPU, container::DEVICE_CPU>;
using syncmem_s2s_d2h_op = container::op::synchronize_memory_op<float, container::DEVICE_CPU, container::DEVICE_GPU>;
using syncmem_d2d_h2h_op = container::op::synchronize_memory_op<double, container::DEVICE_CPU, container::DEVICE_CPU>;
using syncmem_d2d_h2d_op = container::op::synchronize_memory_op<double, container::DEVICE_GPU, container::DEVICE_CPU>;
using syncmem_d2d_d2h_op = container::op::synchronize_memory_op<double, container::DEVICE_CPU, container::DEVICE_GPU>;

using syncmem_c2c_h2h_op = container::op::synchronize_memory_op<std::complex<float>, container::DEVICE_CPU, container::DEVICE_CPU>;
using syncmem_c2c_h2d_op = container::op::synchronize_memory_op<std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
using syncmem_c2c_d2h_op = container::op::synchronize_memory_op<std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;
using syncmem_z2z_h2h_op = container::op::synchronize_memory_op<std::complex<double>, container::DEVICE_CPU, container::DEVICE_CPU>;
using syncmem_z2z_h2d_op = container::op::synchronize_memory_op<std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
using syncmem_z2z_d2h_op = container::op::synchronize_memory_op<std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;

using castmem_s2d_h2h_op = container::op::cast_memory_op<double, float, container::DEVICE_CPU, container::DEVICE_CPU>;
using castmem_s2d_h2d_op = container::op::cast_memory_op<double, float, container::DEVICE_GPU, container::DEVICE_CPU>;
using castmem_s2d_d2h_op = container::op::cast_memory_op<double, float, container::DEVICE_CPU, container::DEVICE_GPU>;
using castmem_d2s_h2h_op = container::op::cast_memory_op<float, double, container::DEVICE_CPU, container::DEVICE_CPU>;
using castmem_d2s_h2d_op = container::op::cast_memory_op<float, double, container::DEVICE_GPU, container::DEVICE_CPU>;
using castmem_d2s_d2h_op = container::op::cast_memory_op<float, double, container::DEVICE_CPU, container::DEVICE_GPU>;

using castmem_c2z_h2h_op = container::op::cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_CPU>;
using castmem_c2z_h2d_op = container::op::cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
using castmem_c2z_d2h_op = container::op::cast_memory_op<std::complex<double>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;
using castmem_z2c_h2h_op = container::op::cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_CPU>;
using castmem_z2c_h2d_op = container::op::cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
using castmem_z2c_d2h_op = container::op::cast_memory_op<std::complex<float>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;

static container::DEVICE_CPU * cpu_ctx = {};
static container::DEVICE_GPU * gpu_ctx = {};

#endif // CONTAINER_MEMORY_H_