/**
 * @file tensor_types.h
 * @brief This file contains the definition of the DataType enum class.
 */
#ifndef MODULE_BASE_CONTAINER_TENSOR_TYPES_H_
#define MODULE_BASE_CONTAINER_TENSOR_TYPES_H_

#include <iostream>
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
 * @brief The type of memory used by an allocator.
 */
enum class AllocatorMemoryType {
    MT_UNKNOWN = 0,  ///< Memory type is unknown.
    MT_CPU = 2,      ///< Memory type is CPU.
    MT_GPU = 1,     ///< Memory type is GPU(CUDA or ROCM).
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
 * Prints the memory type of the enum type AllocatorMemoryType.
 *
 * @param os The output stream to write to.
 * @param tensor The Tensor object to print.
 *
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, const AllocatorMemoryType& memory_type);

#endif // MODULE_BASE_CONTAINER_TENSOR_TYPES_H_