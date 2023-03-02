list(APPEND device_srcs
    memory_op.cpp
)

if(CONTAINER_ENABLE_CUDA_TOOLKIT)
    list(APPEND device_srcs
        cuda/memory_op.cu
    )
endif()

if(CONTAINER_ENABLE_ROCM_TOOLKIT)
    hip_add_library(device_rocm STATIC
        rocm/memory_op.hip.cu
    )
endif()

add_library(device OBJECT ${device_srcs})