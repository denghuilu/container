list(APPEND source_srcs
    tensor.cpp
    cpu_allocator.cpp
    tensor_buffer.cpp
    tensor_shape.cpp
    tensor_types.cpp
)

if(ENABLE_CUDA_TOOLKIT)
    list(APPEND source_srcs gpu_allocator.cpp)
endif()

add_library(source OBJECT ${source_srcs})

add_subdirectory(kernels)