cmake_minimum_required(VERSION 3.20)

project(container
    VERSION 0.0.0
    DESCRIPTION "Container with multi-device, multi-precision support."
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)

add_executable(
    container
    main.cpp
    tensor.cpp
    cpu_allocator.cpp
    tensor_buffer.cpp
    tensor_shape.cpp
    tensor_types.cpp
)