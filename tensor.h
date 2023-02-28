#ifndef MODULE_BASE_CONTAINER_TENSOR_H
#define MODULE_BASE_CONTAINER_TENSOR_H

#include <complex>

#include "allocator.h"
#include "tensor_types.h"
#include "tensor_shape.h"
#include "tensor_buffer.h"

namespace ABACUS {

/**
 * @brief A multi-dimensional array of elements of a single data type.
 *
 * This class represents a Tensor, which is a fundamental concept in ABACUS.
 * A Tensor has a data type, shape, and memory buffer that stores the actual data.
 *
 * This class is not thread-safe and should not be accessed by multiple threads
 * concurrently.
 */
class Tensor {
  public:
    /**
     * @brief Constructor that creates a tensor with the given data type and shape using the default allocator.
     *
     * @param data_type The data type of the tensor.
     * @param shape The shape of the tensor.
     */
    Tensor(DataType data_type, const TensorShape& shape);

    /**
     * @brief Constructor that creates a tensor with the given data pointer, data type and shape.
     *
     * This tensor does not own memory.
     *
     * @param data The data pointer.
     * @param data_type The data type of the tensor.
     * @param shape The shape of the tensor.
     */
    Tensor(void * data, DataType data_type, const TensorShape& shape);

    /**
     * @brief Construct a new Tensor object with the given data type and shape.
     *
     * The memory for the tensor is allocated using the given allocator.
     *
     * @param data_type The data type of the tensor.
     * @param shape The shape of the tensor.
     * @param allocator The allocator to use for allocating memory for the tensor.
     */
    Tensor(DataType data_type, const TensorShape& shape, Allocator* allocator);

    /**
     * @brief Construct a new Tensor object by copying another Tensor.
     *
     * This constructor performs a deep copy of the data buffer of the other tensor.
     *
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor& other);

    /**
     * @brief Get the data type of the tensor.
     *
     * @return The data type of the tensor.
     */
    DataType data_type() const;

    /**
     * @brief Get the shape of the tensor.
     *
     * @return The shape of the tensor.
     */
    const TensorShape& shape() const;

    /**
     * @brief Get the total number of elements in the tensor.
     *
     * @return The total number of elements in the tensor.
     */
    int64_t NumElements() const;

    /**
     * @brief Get a pointer to the data buffer of the tensor.
     *
     * @return A void pointer to the data buffer of the tensor.
     */
    void* data() const;

    /**
     * @brief Get a typed pointer to the data buffer of the tensor.
     *
     * @tparam T The data type of the pointer to return.
     *
     * @return A typed pointer to the data buffer of the tensor.
     *
     * @note The template parameter `T` must match the data type of the tensor. If `T`
     * does not match the data type of the tensor, the behavior is undefined.
     *
     * @note This function returns a pointer to the first element in the data buffer
     * of the tensor. If the tensor is empty, the behavior is undefined.
     */
    template <typename T>
    T* data() const {
        if ((std::is_same<T, float>::value && data_type_ != DataType::DT_FLOAT) ||
            (std::is_same<T, int>::value && data_type_ != DataType::DT_INT) ||
            (std::is_same<T, int64_t>::value && data_type_ != DataType::DT_INT64) ||
            (std::is_same<T, double>::value && data_type_ != DataType::DT_DOUBLE) ||
            (std::is_same<T, std::complex<float>>::value && data_type_ != DataType::DT_COMPLEX) ||
            (std::is_same<T, std::complex<double>>::value && data_type_ != DataType::DT_COMPLEX_DOUBLE))
        {
            std::cerr << "Tensor data type does not match requested type." << std::endl;
            exit(EXIT_FAILURE);
        }
        return buffer_.base<T>();
    }


    /**
     * @brief Get the TensorBuffer object that holds the data of the tensor.
     *
     * @return The TensorBuffer object that holds the data of the tensor.
     */
    const TensorBuffer& buffer() const;

    /**
     * @brief Returns the size of a single element for a given data type.
     *
     * @param dtype The data type.
     *
     * @return The size of a single element.
     *
     * @note This function is not thread-safe.
     * @note If an unsupported data type is passed in, an error message will be printed and the program will exit.
     * @note This function returns the size in bytes of a single element of the given data type.
     * If an unsupported data type is provided, the function outputs an error message and exits the program.
     * Supported data types:
     * DT_FLOAT: 4 bytes
     * DT_INT32: 4 bytes
     * DT_DOUBLE: 8 bytes
     * DT_COMPLEX: 8 bytes (2 floats)
     * DT_COMPLEX_DOUBLE: 16 bytes (2 doubles)
     */
    static size_t SizeOfType(DataType dtype) {
        switch (dtype) {
            case DataType::DT_FLOAT:
                return sizeof(float);
            case DataType::DT_INT:
                return sizeof(int32_t);
            case DataType::DT_DOUBLE:
                return sizeof(double);
            case DataType::DT_COMPLEX:
                return sizeof(std::complex<float>);
            case DataType::DT_COMPLEX_DOUBLE:
                return sizeof(std::complex<double>);
            default:
                std::cerr << "Unsupported data type!" << std::endl;
                exit(EXIT_FAILURE);
        }
    }

private:
    /**
     * @brief The data type of the tensor.
     */
    DataType data_type_;

    /**
     * @brief The shape of the tensor.
     */
    TensorShape shape_;

    /**
     * @brief The allocator used to allocate the memory for the tensor.
     */
    Allocator* allocator_;

    /**
     * @brief The TensorBuffer object that holds the data of the tensor.
     */
    TensorBuffer buffer_;


};

/**
 * @brief Overloaded operator<< for the Tensor class.
 *
 * Prints the data of the Tensor object to the given output stream.
 *
 * @param os The output stream to write to.
 * @param tensor The Tensor object to print.
 *
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

}
#endif // MODULE_BASE_CONTAINER_TENSOR_H
