#include "tensor_types.h"

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
std::ostream& operator<<(std::ostream& os, const DataType& data_type) {
    switch (data_type) {
        case DataType::DT_FLOAT:
            os << "float";
            break;
        case DataType::DT_DOUBLE:
            os << "float64";
            break;
        case DataType::DT_INT:
            os << "int32";
            break;
        case DataType::DT_INT64:
            os << "int64";
            break;
        case DataType::DT_COMPLEX:
            os << "complex<float>";
            break;
        case DataType::DT_COMPLEX_DOUBLE:
            os << "complex<double>";
            break;
        default:
            os << "unknown";
            break;
    }
    return os;
}

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
std::ostream& operator<<(std::ostream& os, const AllocatorMemoryType& memory_type) {
    switch (memory_type) {
        case AllocatorMemoryType::MT_CPU:
            os << "cpu";
            break;
        case AllocatorMemoryType::MT_GPU:
            os << "gpu";
            break;
        default:
            os << "unknown";
            break;
    }
    return os;
}