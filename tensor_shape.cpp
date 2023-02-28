#include "tensor_shape.h"

namespace ABACUS {
/**
 * @brief Namespace containing constants for default constructor
 */
namespace {
    /**
     * @brief Default size of a dimension
     */
    constexpr int kDefaultDimSize = 0;
} // namespace

/**
 * @brief Default constructor for TensorShape class
 * Initializes TensorShape with default dimensions
 */
TensorShape::TensorShape() : dims_(kDefaultDimSize) {}

/**
 * @brief Constructor for TensorShape class
 * @param dims an initializer list of integers to initialize TensorShape dimensions
 */
TensorShape::TensorShape(std::initializer_list<int> dims) : dims_(dims) {}

/**
 * @brief Constructor for TensorShape class
 * @param dims a vector of integers to initialize TensorShape dimensions
 */
TensorShape::TensorShape(const std::vector<int>& dims) : dims_(dims) {}

/**
 * @brief Copy constructor for TensorShape class
 * @param other another instance of TensorShape to copy from
 */
TensorShape::TensorShape(const TensorShape& other) = default;

/**
 * @brief Get size of a specific dimension in the tensor
 * @param dim index of the dimension to get size from
 * @return size of the specified dimension
 */
int TensorShape::dim_size(int dim) const {
    return dims_[dim];
}

/**
 * @brief Get all dimension sizes in the tensor
 * @return a constant reference to the vector containing all dimension sizes
 */
const std::vector<int>& TensorShape::dims() const {
    return dims_;
}

/**
 * @brief Get rank of the tensor, i.e., number of dimensions
 * @return the rank of the tensor
 */
unsigned int TensorShape::ndims() const {
    return dims_.size();
}

/**
* @brief Returns the total number of elements in the shape.
*
* @return int64_t The number of elements.
*/
int64_t TensorShape::NumElements() const {
    int64_t num_elements = 1;
    for (int i = 0; i < this->ndims(); ++i) {
        num_elements *= dims_[i];
    }
    return num_elements;
}

/**
 * @brief Modify size of a specific dimension in the tensor
 * @param dim index of the dimension to modify size
 * @param size new size of the specified dimension
 */
void TensorShape::set_dim_size(int dim, int size) {
    dims_[dim] = size;
}

/**
 * @brief Add a new dimension to the tensor
 * @param size size of the new dimension to add
 */
void TensorShape::add_dim(int size) {
    dims_.push_back(size);
}

/**
 * @brief Remove a dimension from the tensor
 * @param dim index of the dimension to remove
 */
void TensorShape::remove_dim(int dim) {
    dims_.erase(dims_.begin() + dim);
}

/**
 * @brief Overload the == operator to compare two TensorShape objects
 * @param other the TensorShape object to compare to
 * @return true if the two objects are equal, false otherwise
 */
bool TensorShape::operator==(const TensorShape& other) const {
    return dims_ == other.dims_;
}

/**
 * @brief Overload the != operator to compare two TensorShape objects
 * @param other the TensorShape object to compare to
 * @return true if the two objects are not equal, false otherwise
 */
bool TensorShape::operator!=(const TensorShape& other) const {
    return dims_ != other.dims_;
}

/**
 * @brief Overload the << operator to print the tensor shape
 * @param os the output stream to write to
 * @param shape the TensorShape object to print
 * @return a reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
    os << "[";
    for (int i = 0; i < shape.ndims(); ++i) {
        os << shape.dims()[i];
        if (i < shape.ndims() - 1) {
            os << ",";
        }
    }
    os << "]";
    return os;
}

}