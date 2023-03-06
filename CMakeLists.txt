cmake_minimum_required(VERSION 3.18)

project(container
    VERSION 0.0.0
    DESCRIPTION "Container with multi-device, multi-precision support."
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option (ENABLE_CUDA_TOOLKIT "Enable support to CUDA for container." OFF)

set(CMAKE_CXX_STANDARD 11)

add_executable(container source/main.cpp)

if(ENABLE_CUDA_TOOLKIT)
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

if(DEFINED ENV{MKLROOT} AND NOT DEFINED MKLROOT)
    set(MKLROOT "$ENV{MKLROOT}")
endif()
if(MKLROOT)
    find_package(IntelMKL REQUIRED)
    add_definitions(-D__MKL)
    include_directories(${MKL_INCLUDE_DIRS} ${MKL_INCLUDE_DIRS}/fftw)

    # Since libtorch will find its own MKL, the fftw part conflicts with the original one.
    # When enable deepks, mkl will be linked within ${TORCH_LIBRARIES}.
    if(NOT ENABLE_DEEPKS)
        list(APPEND math_libs IntelMKL::MKL)
    endif()
else()
    find_package(FFTW3 REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(ScaLAPACK REQUIRED)
    include_directories(${FFTW3_INCLUDE_DIRS})
    list(APPEND math_libs
        FFTW3::FFTW3
        LAPACK::LAPACK
        ScaLAPACK::ScaLAPACK
    )
    if(USE_OPENMP)
        list(APPEND math_libs FFTW3::FFTW3_OMP)
    endif()
    if (ENABLE_FLOAT_FFTW)
        list(APPEND math_libs FFTW3::FFTW3_FLOAT)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
        list(APPEND math_libs -lgfortran)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
        list(APPEND math_libs -lifcore)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
        list(APPEND math_libs -lgfortran)
    else()
        message(WARNING "Cannot find the correct library for Fortran.")
    endif()
endif()

if (ENABLE_FLOAT_FFTW)
    add_definitions(-D__ENABLE_FLOAT_FFTW)
endif()

add_subdirectory(source)

target_link_libraries(
    container
    PRIVATE
    source
    device
)