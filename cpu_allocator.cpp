#include <cstdlib> // for ::operator new, ::operator delete
#include <cassert> // for assert

#include "cpu_allocator.h"

namespace ABACUS {

/**
 * @brief Allocate a block of CPU memory with the given size and default alignment.
 *
 * @param size The size of the memory block to allocate.
 *
 * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
 */
void* CPUAllocator::allocate(size_t size) {
    return ::operator new(size);
}

/**
 * @brief Allocate a block of CPU memory with the given size and alignment.
 *
 * @param size The size of the memory block to allocate.
 * @param alignment The alignment of the memory block to allocate.
 *
 * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
 */
void* CPUAllocator::allocate(size_t size, size_t alignment) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
    return ptr;
}

/**
 * @brief Free a block of CPU memory that was previously allocated by this allocator.
 *
 * @param ptr A pointer to the memory block to free.
 */
void CPUAllocator::free(void* ptr) {
    ::operator delete(ptr);
}

/**
 * @brief Get the allocated size of a given pointer.
 *
 * @param ptr The pointer to get the allocated size of.
 * @return size_t The size of the allocated block of memory, in bytes.
 *
 * @note This function is not implemented for CPUAllocator and always returns 0.
 */
size_t CPUAllocator::AllocatedSize(void* ptr) {
    assert(false && "not implemented");
    return 0;
}

/**
 * @brief Get the type of memory used by the TensorBuffer.
 *
 * @return AllocatorMemoryType The type of memory used by the TensorBuffer.
 */
AllocatorMemoryType CPUAllocator::GetMemoryType() {
    return AllocatorMemoryType::MT_CPU;
}

} // namespace ABACUS
