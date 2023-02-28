#include "tensor_buffer.h"

namespace ABACUS {

/**
 * @brief Construct a new TensorBuffer object.
 *
 * @param alloc Pointer to the allocator used to allocate the buffer.
 * @param data_ptr Pointer to the underlying data buffer.
 */
TensorBuffer::TensorBuffer(Allocator* alloc, void* data_ptr) : data_(data_ptr), alloc_(alloc), owns_memory(true) {}

/**
  * @brief Construct a new TensorBuffer object.
  *
  * This is a reference TensorBuffer, does not owns memory itself.
  *
  * @param data_ptr Pointer to the given data buffer.
  */
TensorBuffer::TensorBuffer(void* data_ptr) : data_(data_ptr), alloc_(), owns_memory(false) {}

/**
 * @brief Destroy the TensorBuffer object.
 */
TensorBuffer::~TensorBuffer() {
    if (this->OwnsMemory()) {
        alloc_->free(data_);
    }
}

/**
 * @brief Get the raw data pointer.
 *
 * @return void* Pointer to the underlying data buffer.
 */
void* TensorBuffer::data() const { return data_; }

/**
 * @brief Get the total number of bytes allocated for the buffer.
 *
 * This method returns the total number of bytes allocated for the buffer by the allocator
 * associated with the TensorBuffer. If the buffer is not yet allocated, the function returns 0.
 *
 * @return The total number of bytes allocated for the buffer.
 */
size_t TensorBuffer::GetAllocatedBytes() const {
    return alloc_ == nullptr ?
           0 :
           alloc_->AllocatedSize(data());
}

/**
 * @brief Get the root TensorBuffer object.
 *
 * If this TensorBuffer is a sub-buffer of another TensorBuffer, returns that
 * TensorBuffer. Otherwise, returns this.
 *
 * @return TensorBuffer* Pointer to the root TensorBuffer object.
 */
TensorBuffer* TensorBuffer::root_buffer() { return this; } // Implementation goes here.

/**
 * @brief Get the Allocator object used in this class.
 *
 * @return Allocator* Pointer to the Allocator object.
 */
    Allocator * TensorBuffer::allocator() const {
    return alloc_;
}

/**
 * @brief Check whether this TensorBuffer owns the underlying memory.
 *
 * @return true If the TensorBuffer owns the underlying memory.
 * @return false If the TensorBuffer does not own the underlying memory.
 */
bool TensorBuffer::OwnsMemory() const { return this->owns_memory; }

/**
 * @brief Get the type of memory used by the TensorBuffer.
 *
 * @return AllocatorMemoryType The type of memory used by the TensorBuffer.
 */
AllocatorMemoryType TensorBuffer::GetMemoryType() const {
    if (alloc_ != nullptr) {
        return alloc_->GetMemoryType();
    }
    return AllocatorMemoryType::MT_UNKNOWN;
}

}  // namespace ABACUS
