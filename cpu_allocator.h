#ifndef MODULE_BASE_CONTAINER_CPU_ALLOCATOR_H
#define MODULE_BASE_CONTAINER_CPU_ALLOCATOR_H

#include "allocator.h"

namespace container {

/**
 * @brief An Allocator subclass for CPU memory.
 *
 * This class provides an implementation of the Allocator interface for CPU memory. It
 * uses the standard library functions std::malloc, std::free, and std::aligned_alloc
 * to allocate and deallocate memory blocks.
 */
class CPUAllocator : public Allocator {
  public:
    /**
     * @brief Allocate a block of CPU memory with the given size and default alignment.
     *
     * @param size The size of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    void* allocate(size_t size) override;

    /**
     * @brief Allocate a block of CPU memory with the given size and alignment.
     *
     * @param size The size of the memory block to allocate.
     * @param alignment The alignment of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    void* allocate(size_t size, size_t alignment) override;

    /**
     * @brief Free a block of CPU memory that was previously allocated by this allocator.
     *
     * @param ptr A pointer to the memory block to free.
     */
    void free(void* ptr) override;

    /**
     * @brief Get the allocated size of a given pointer.
     *
     * @param ptr The pointer to get the allocated size of.
     * @return size_t The size of the allocated block of memory, in bytes.
     *
     * @note This function is not implemented for CPUAllocator and always returns 0.
     */
    size_t AllocatedSize(void* ptr) override;

    /**
     * @brief Get the type of memory used by the TensorBuffer.
     *
     * @return AllocatorMemoryType The type of memory used by the TensorBuffer.
     */
    AllocatorMemoryType GetMemoryType() override;
};

} // namespace container

#endif // MODULE_BASE_CONTAINER_CPU_ALLOCATOR_H
