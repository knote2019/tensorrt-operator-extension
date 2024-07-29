#-----------------------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(tensorrt_operator_extension LANGUAGES CXX VERSION 1.0.0)

# host information.
message("==================================")
cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
message("hostname: ${HOSTNAME}")
message("rootpath: ${PROJECT_SOURCE_DIR}")
message("==================================")

# c++ settings.
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fpic -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_BUILD_TYPE Debug)

# cmake settings.
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

# cuda settings.
set(CMAKE_CUDA_COMPILER /usr/bin/clang++)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "-std=c++17")
set(CMAKE_CUDA_ARCHITECTURES "80")
# enable after cuda compiler set.
enable_language(CUDA)

#-----------------------------------------------------------------------------------------------------------------------
# cuda.
set(CUDA_INCLUDE /usr/local/cuda/include)
find_library(CUDART_LIBRARY cudart /usr/local/cuda/lib64)
message("==================================")
message("CUDA_INCLUDE=${CUDA_INCLUDE}")
message("CUDART_LIBRARY=${CUDART_LIBRARY}")
message("==================================")

# tensorrt.
set(TENSORRT_INCLUDE /usr/local/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer /usr/local/tensorrt/lib)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin /usr/local/tensorrt/lib)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser /usr/local/tensorrt/lib)
message("==================================")
message("TENSORRT_INCLUDE=${TENSORRT_INCLUDE}")
message("TENSORRT_LIBRARY=${TENSORRT_LIBRARY}")
message("TENSORRT_PLUGIN_LIBRARY=${TENSORRT_PLUGIN_LIBRARY}")
message("TENSORRT_ONNX_LIBRARY=${TENSORRT_ONNX_LIBRARY}")
message("==================================")

#-----------------------------------------------------------------------------------------------------------------------
# sources.
SET(TEST_REGEX "src/*.cpp" CACHE STRING "TEST_REGEX")
if ("${TEST_REGEX}" MATCHES ".*.cpp")
    file(GLOB_RECURSE TEST_SRC ${TEST_REGEX})
else ()
    file(GLOB_RECURSE TEST_SRC ${TEST_REGEX}/*.cpp)
endif ()
file(GLOB_RECURSE CUDA_SRCS src/*.cu)

# set target.
add_executable(runTests)
target_include_directories(runTests PUBLIC
        gtest
        gtest/include
        include
        ${CUDA_INCLUDE}
        ${TENSORRT_INCLUDE}
)
target_sources(runTests PUBLIC
        gtest/src/gtest_main.cc
        gtest/src/gtest-all.cc
        ${TEST_SRC}
        ${CUDA_SRCS}
)
target_link_libraries(runTests PUBLIC
        ${CUDART_LIBRARY}
        ${TENSORRT_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY}
        ${TENSORRT_ONNX_LIBRARY}
)
