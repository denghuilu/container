#include <iostream>
#include <cstring>
#include <complex>

#include "tensor.h"
#include "cpu_allocator.h"

namespace container {

// Constructor that creates a tensor with the given data type and shape using the default allocator.
Tensor::Tensor(DataType data_type, const TensorShape& shape)
        : data_type_(data_type),
          shape_(shape),
          allocator_(new CPUAllocator()),
          buffer_(allocator_, allocator_->allocate(shape.NumElements() * SizeOfType(data_type))) {}

// Constructor that creates a tensor with the given data pointer, data type and shape.
Tensor::Tensor(void *data, DataType data_type, const TensorShape &shape)
        : data_type_(data_type),
          shape_(shape),
          allocator_(),
          buffer_(data) {}

// Construct a new Tensor object with the given data type and shape.
Tensor::Tensor(DataType data_type, const TensorShape& shape, Allocator* allocator)
        : data_type_(data_type),
          shape_(shape),
          allocator_(allocator),
          buffer_(allocator_, allocator_->allocate(shape.NumElements() * SizeOfType(data_type))) {}

// Construct a new Tensor object by copying another Tensor.
Tensor::Tensor(const Tensor& other)
        : data_type_(other.data_type_),
          shape_(other.shape_),
          allocator_(other.allocator_),
          buffer_(allocator_, allocator_->allocate(shape_.NumElements() * SizeOfType(data_type_))) {
    std::memcpy(buffer_.data(), other.data(), buffer_.GetAllocatedBytes());
}

// Get the data type of the tensor.
DataType Tensor::data_type() const { return data_type_; }

// Get the shape of the tensor.
const TensorShape& Tensor::shape() const { return shape_; }

// Get the total number of elements in the tensor.
int64_t Tensor::NumElements() const { return shape_.NumElements(); }

// Get a pointer to the data buffer of the tensor.
void* Tensor::data() const { return buffer_.data(); }

// Get the TensorBuffer object that holds the data of the tensor.
const TensorBuffer& Tensor::buffer() const { return buffer_; }


// Overloaded operator<< for the Tensor class.
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    const int64_t num_elements = tensor.NumElements();
    const DataType data_type = tensor.data_type();
    const TensorShape& shape = tensor.shape();
    os << "Tensor(";
    os << "shape=(";
    for (int i = 0; i < shape.ndims(); ++i) {
        os << shape.dim_size(i);
        if (i < shape.ndims() - 1) {
            os << ",";
        }
    }
    os << "), data_type=" << tensor.data_type();
    os << ", buffer=[";
    switch (data_type) {
        case DataType::DT_FLOAT: {
            const auto* data = static_cast<const float*>(tensor.data());
            for (int64_t i = 0; i < num_elements; ++i) {
                os << data[i];
                if (i < num_elements - 1) {
                    os << ",";
                }
            }
            break;
        }
        case DataType::DT_DOUBLE: {
            const auto* data = static_cast<const double*>(tensor.data());
            for (int64_t i = 0; i < num_elements; ++i) {
                os << data[i];
                if (i < num_elements - 1) {
                    os << ",";
                }
            }
            break;
        }
        case DataType::DT_INT: {
            const auto* data = static_cast<const int*>(tensor.data());
            for (int64_t i = 0; i < num_elements; ++i) {
                os << data[i];
                if (i < num_elements - 1) {
                    os << ",";
                }
            }
            break;
        }
        case DataType::DT_INT64: {
            const auto* data = static_cast<const int64_t*>(tensor.data());
            for (int64_t i = 0; i < num_elements; ++i) {
                os << data[i];
                if (i < num_elements - 1) {
                    os << ",";
                }
            }
            break;
        }
        case DataType::DT_COMPLEX: {
            const auto* data = static_cast<const std::complex<float>*>(tensor.data());
            for (int64_t i = 0; i < num_elements; ++i) {
                os << "{" << data[i].real() << ", " << data[i].imag() << "}";
                if (i < num_elements - 1) {
                    os << ",";
                }
            }
            break;
        }
        case DataType::DT_COMPLEX_DOUBLE: {
            const auto* data = static_cast<const std::complex<double>*>(tensor.data());
            for (int64_t i = 0; i < num_elements; ++i) {
                os << "{" << data[i].real() << ", " << data[i].imag() << "}";
                if (i < num_elements - 1) {
                    os << ",";
                }
            }
            break;
        }
        default:
            os << "unknown";
            break;
    }
    os << "])";
    return os;
}

} // namespace container
