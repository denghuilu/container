#include <cstdlib> // for ::operator new, ::operator delete
#include <cassert> // for assert
#include <cuda_runtime.h> // for CUDA APIs

#include "gpu_allocator.h"

namespace ABACUS {

/**
 * @brief Allocate a block of memory with the given size and default alignment on GPU.
 *
 * @param size The size of the memory block to allocate.
 *
 * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
 */
void* GPUAllocator::allocate(size_t size) {
    void* ptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

/**
 * @brief Allocate a block of CPU memory with the given size and alignment.
 *
 * @param size The size of the memory block to allocate.
 * @param alignment The alignment of the memory block to allocate.
 *
 * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
 */
void* GPUAllocator::allocate(size_t size, size_t alignment) {
    void* ptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

/**
 * @brief Free a block of CPU memory that was previously allocated by this allocator.
 *
 * @param ptr A pointer to the memory block to free.
 */
void GPUAllocator::free(void* ptr) {
    cudaFree(ptr);
}

/**
 * @brief Get the allocated size of a given pointer.
 *
 * @param ptr The pointer to get the allocated size of.
 * @return size_t The size of the allocated block of memory, in bytes.
 *
 * @note This function is not implemented for GPUAllocator and always returns 0.
 */
size_t GPUAllocator::AllocatedSize(void* ptr) {
    assert(false && "not implemented");
    return 0;
}

/**
 * @brief Get the type of memory used by the TensorBuffer.
 *
 * @return AllocatorMemoryType The type of memory used by the TensorBuffer.
 */
AllocatorMemoryType GPUAllocator::GetMemoryType() {
    return AllocatorMemoryType::MT_GPU;
}

} // namespace ABACUS
