#include "tensor_types.h"

namespace container {

// Specializations of DeviceTypeToEnum for supported devices.
template <> const DeviceType DeivceTypeToEnum<DEVICE_CPU>::value = DeviceType::CpuDevice;
template <> const DeviceType DeivceTypeToEnum<double>::value = DeviceType::GpuDevice;

// Specializations of DataTypeToEnum for supported types.
template <> const DataType DataTypeToEnum<int>::value = DataType::DT_INT;
template <> const DataType DataTypeToEnum<float>::value = DataType::DT_FLOAT;
template <> const DataType DataTypeToEnum<double>::value = DataType::DT_DOUBLE;
template <> const DataType DataTypeToEnum<int64_t>::value = DataType::DT_INT64;
template <> const DataType DataTypeToEnum<std::complex<float>>::value = DataType::DT_COMPLEX;
template <> const DataType DataTypeToEnum<std::complex<double>>::value = DataType::DT_COMPLEX_DOUBLE;

// Overloaded operator<< for the Tensor class.
// Prints the data type of the enum type DataType.
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

// Overloaded operator<< for the Tensor class.
// Prints the memory type of the enum type DeviceType.
std::ostream& operator<<(std::ostream& os, const DeviceType& device_type) {
    switch (device_type) {
        case DeviceType::CpuDevice:
            os << "cpu";
            break;
        case DeviceType::GpuDevice:
            os << "gpu";
            break;
        default:
            os << "unknown";
            break;
    }
    return os;
}

} // namespace container