#ifndef CONTAINER_TENSOR_MAP_H
#define CONTAINER_TENSOR_MAP_H

#include <complex>

#include "tensor.h"

namespace container {

/**
 * @brief A multi-dimensional reference array of elements of a single data type.
 *
 * This class represents a Tensor, which is a fundamental concept in container.
 * A Tensor has a data type, shape, and memory buffer that stores the actual data.
 *
 * This class is not thread-safe and should not be accessed by multiple threads
 * concurrently.
 */
 class TensorMap : public Tensor {
 public:

    /**
     * @brief Constructor that map the given data pointer to a tensor object with the given
     * data type, device type and shape.
     *
     * This tensor does not own memory.
     *
     * @param data The data pointer.
     * @param data_type The data type of the tensor.
     * @param device The data type of the tensor.
     * @param shape The shape of the tensor.
     */
    TensorMap(void *data, DataType data_type, DeviceType device, const TensorShape &shape);
};

} // namespace container

#endif // CONTAINER_TENSOR_MAP_H
