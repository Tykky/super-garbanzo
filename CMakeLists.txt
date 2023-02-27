set(CMAKE_CUDA_ARCHITECTURES 75)

cmake_minimum_required(VERSION 3.8)
project(super-garbanzo LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(cuda)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)