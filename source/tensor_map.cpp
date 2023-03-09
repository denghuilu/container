#include "tensor_map.h"

namespace container {
// Constructor that creates a tensor with the given data pointer, data type, device type and shape.
TensorMap::TensorMap(void *data, DataType data_type, DeviceType device, const TensorShape &shape)
        : Tensor(data, data_type, device, shape) {}

} // namespace container