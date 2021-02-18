cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(porting-cuda-to-sycl LANGUAGES CXX)

if (NOT SYCL_ROOT)
   set(SYCL_ROOT $ENV{SYCL_ROOT})
endif()

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "SYCL_ROOT not specified! Please specify -DSYCL_ROOT=<DPCPP_ROOT_PATH>")
endif()

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(CMAKE_CXX_COMPILER "${SYCL_ROOT}/bin/clang++") 
set(SYCL_FLAGS 
      "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
      "-fsycl-unnamed-lambda"
      "-std=c++17")

set(EXTRA_FLAGS "-DROOT_NO_INT128")
    
add_executable(test1 test1.cpp)
target_include_directories(test1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
  )
target_compile_features(test1 PUBLIC cxx_std_17)
target_compile_options(test1 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
target_link_libraries(test1 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
target_include_directories(test1 PUBLIC ${SYCL_INCLUDE_DIR})

add_executable(test2 test2.cpp)
target_include_directories(test2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
  )
target_compile_features(test2 PUBLIC cxx_std_17)
target_compile_options(test2 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
target_link_libraries(test2 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
target_include_directories(test2 PUBLIC ${SYCL_INCLUDE_DIR})

