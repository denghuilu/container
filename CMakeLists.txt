cmake_minimum_required(VERSION 3.16)
if(POLICY CMP0135) # https://cmake.org/cmake/help/git-stage/policy/CMP0135.html
    cmake_policy(SET CMP0135 NEW) # Otherwise this policy generates a warning on CMake 3.24
endif()

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