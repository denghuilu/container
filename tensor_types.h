/**
 * @file tensor_types.h
 * @brief This file contains the definition of the DataType enum class.
 */
#ifndef MODULE_BASE_CONTAINER_TENSOR_TYPES_H_
#define MODULE_BASE_CONTAINER_TENSOR_TYPES_H_

#include <complex>
#include <iostream>

namespace container {

/**
@brief Enumeration of data types for tensors.
The DataType enum lists the supported data types for tensors. Each data type
is identified by a unique value. The DT_INVALID value is reserved for invalid
data types.
*/
enum class DataType {
    DT_INVALID = 0, /**< Invalid data type */
    DT_FLOAT = 1, /**< Single-precision floating point */
    DT_DOUBLE = 2, /**< Double-precision floating point */
    DT_INT = 3, /**< 32-bit integer */
    DT_INT64 = 4, /**< 64-bit integer */
    DT_COMPLEX = 5, /**< 32-bit complex */
    DT_COMPLEX_DOUBLE = 6, /**< 64-bit complex */
// ... other data types
};

/**
 *@struct DEVICE_CPU, DEVICE_GPU
 *@brief A tag type for identifying CPU and GPU devices.
*/
struct DEVICE_CPU;
struct DEVICE_GPU;

/**
 * @brief The type of memory used by an allocator.
 */
enum class DeviceType {
    UnKnown = 0,  ///< Memory type is unknown.
    CpuDevice = 1,      ///< Memory type is CPU.
    GpuDevice = 2,     ///< Memory type is GPU(CUDA or ROCm).
};

/**
 * @brief Template struct for mapping a Device Type to its corresponding enum value.
 *
 * @param T The DataType to map to its enum value.
 *@return The enumeration value corresponding to the data type.
 *  This method uses template specialization to map each supported data type to its
 *          corresponding enumeration value. If the template argument T is not a supported
 *  data type, this method will cause a compile-time error.
 *  Example usage:
 *      DataTypeToEnum<float>::value; // Returns DataType::DT_FLOAT
 */
template <typename T>
struct DeivceTypeToEnum {
    static const DeviceType value;
};

/**
 * @brief Template struct for mapping a DataType to its corresponding enum value.
 *
 * @param T The DataType to map to its enum value.
 *@return The enumeration value corresponding to the data type.
 *  This method uses template specialization to map each supported data type to its
 *          corresponding enumeration value. If the template argument T is not a supported
 *  data type, this method will cause a compile-time error.
 *  Example usage:
 *      DataTypeToEnum<float>::value; // Returns DataType::DT_FLOAT
 */
 template <typename T>
 struct DataTypeToEnum {
     static const DataType value;
 };

/**
 * @brief Overloaded operator<< for the Tensor class.
 *
 * Prints the data type of the enum type DataType.
 *
 * @param os The output stream to write to.
 * @param tensor The Tensor object to print.
 *
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, const DataType& data_type);

/**
 * @brief Overloaded operator<< for the Tensor class.
 *
 * Prints the memory type of the enum type MemoryType.
 *
 * @param os The output stream to write to.
 * @param tensor The Tensor object to print.
 *
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, const DeviceType& memory_type);

} // namespace container
#endif // MODULE_BASE_CONTAINER_TENSOR_TYPES_H_