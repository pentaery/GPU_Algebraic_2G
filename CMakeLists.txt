cmake_minimum_required(VERSION 3.10)

# 强制使用GCC 12
set(CMAKE_C_COMPILER gcc-12)
set(CMAKE_CXX_COMPILER g++-12)

project(amgx_test LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(eigensolver eigensolver.cc matMFEM.cc)

target_include_directories(eigensolver PRIVATE 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  $ENV{HOME}/HPCSoft/METIS/include
  $ENV{HOME}/HPCSoft/mfem)



target_link_libraries(eigensolver $ENV{HOME}/HPCSoft/METIS/lib/libmetis.so)
target_link_libraries(eigensolver $ENV{HOME}/HPCSoft/GKlib/lib/libGKlib.a)
target_link_libraries(eigensolver $ENV{HOME}/HPCSoft/mfem/libmfem.a)
target_link_libraries(eigensolver CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cusolver)



message(STATUS "CUDA Toolkit found: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

