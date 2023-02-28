cmake_minimum_required(VERSION 3.20)

project(container
    VERSION 0.0.0
    DESCRIPTION "Container with multi-device, multi-precision support."
    LANGUAGES CXX
)

option (CONTAINER_ENABLE_CUDA_TOOLKIT "Enable support to CUDA for container." OFF)

set(CMAKE_CXX_STANDARD 11)

if(CONTAINER_ENABLE_CUDA_TOOLKIT)
    add_executable(
        container
        main.cpp
        tensor.cpp
        cpu_allocator.cpp
        gpu_allocator.cpp
        tensor_buffer.cpp
        tensor_shape.cpp
        tensor_types.cpp
    )
else()
    add_executable(
        container
        main.cpp
        tensor.cpp
        cpu_allocator.cpp
        tensor_buffer.cpp
        tensor_shape.cpp
        tensor_types.cpp
    )
endif()

if(CONTAINER_ENABLE_CUDA_TOOLKIT)
    include(CheckLanguage)
    check_language(CUDA)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set_property(
        TARGET container
        PROPERTY CUDA_ARCHITECTURES
        60 # P100
        70 # V100
        75 # T4
        80 # A100
    )
    target_link_libraries(container PRIVATE
        CUDA::cufft
        CUDA::cudart
        CUDA::cublas
    )
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_compile_definitions(__CUDA)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
    endif()
endif()