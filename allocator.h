#ifndef MODULE_BASE_CONTAINER_ALLOCATOR_H
#define MODULE_BASE_CONTAINER_ALLOCATOR_H

#include <cstddef>

#include "tensor_types.h"

namespace container {

/**
 * @brief An abstract base class for memory allocators.
 *
 * This class defines an interface for memory allocators. Subclasses of this class
 * can provide different implementations of memory allocation/deallocation strategies.
 *
 * All memory allocated by an Allocator must be freed using the same allocator that
 * allocated it.
 */
class Allocator {
  public:
    /**
     * @brief Allocate a block of memory with the given size and default alignment.
     *
     * @param size The size of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Allocate a block of memory with the given size and alignment.
     *
     * @param size The size of the memory block to allocate.
     * @param alignment The alignment of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    virtual void* allocate(size_t size, size_t alignment) = 0;

    /**
     * @brief Free a block of memory that was previously allocated by this allocator.
     *
     * @param ptr A pointer to the memory block to free.
     */
    virtual void free(void* ptr) = 0;

    /**
     * @brief Destroy the Allocator object.
     *
     * This method is responsible for freeing any resources that the allocator has acquired
     * during its lifetime.
     */
    virtual ~Allocator() = default;

    /**
     * @brief Get the allocated size of a given pointer.
     *
     * @param ptr The pointer to get the allocated size of.
     * @return size_t The size of the allocated block of memory, in bytes.
     */
    virtual size_t AllocatedSize(void* ptr) = 0;

    /**
     * @brief Get the type of memory used by the TensorBuffer.
     *
     * @return AllocatorMemoryType The type of memory used by the TensorBuffer.
     */
    virtual AllocatorMemoryType GetMemoryType() = 0;
};

} // namespace ABACUS

#endif // MODULE_BASE_CONTAINER_ALLOCATOR_H
