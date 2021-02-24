cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(porting-cuda-to-sycl LANGUAGES CXX)

if (NOT SYCL_ROOT)
   set(SYCL_ROOT $ENV{SYCL_ROOT})
endif()

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "SYCL_ROOT not specified! Please specify -DSYCL_ROOT=<DPCPP_ROOT_PATH>")
endif()

set(CMAKE_VERBOSE_MAKEFILE 1)

macro(build_tests TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
      $<INSTALL_INTERFACE:base>
      )
    #target_compile_features(${TARGET_NAME}  PUBLIC cxx_std_17)
    target_compile_options(${TARGET_NAME}  PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
    target_link_libraries(${TARGET_NAME}  PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
    target_include_directories(${TARGET_NAME}  PUBLIC ${SYCL_INCLUDE_DIR})
  endforeach()
endmacro()

macro(add_to_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
  endforeach()
endmacro()

set(ONEAPI_UNIT_TESTS_BASE
  test1.cpp                    # converted (unmodified) test_ranluxpp.cu, unit test for RANLUX++
  test2.cpp                    # converted (and modified) test_ranluxpp.cu, unit test for RANLUX++
  test3.cpp                    # converted (and modified) cufisher_price_v2.cu example
  test4.cpp                    # simplified cufisher_price_v2.cu example
  test5.cpp                    # empty program, testing clang++ and the CMakeLists.txt
  test6.cpp                    # converted (and modified) test_atomic.cu 
  test7.cpp                    # converted (and modified) test_atomic.cu 
)

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(CMAKE_C_COMPILER "${SYCL_ROOT}/bin/clang++")
set(CMAKE_CXX_COMPILER "${SYCL_ROOT}/bin/clang++")
set(SYCL_FLAGS
  	"-fsycl"
  	"-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
  	"-fsycl-unnamed-lambda"
  	"-std=c++20"
  	"-stdlib=libstdc++")

#set(EXTRA_FLAGS "-DNDEBUG" "-DROOT_NO_INT128" "-DCOPCORE_CUDA_COMPILER" 
set(EXTRA_FLAGS "-DNDEBUG" "-DROOT_NO_INT128" "-I../../local/include")
#"-DCOPCORE_DEVICE_COMPILATION" )

build_tests("${ONEAPI_UNIT_TESTS_BASE}")
add_to_test("${ONEAPI_UNIT_TESTS_BASE}")

# Specific options for test_sparsevector
#set_target_properties(test_sparsevector PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#target_compile_options(test_sparsevector PRIVATE -rdc=true;-lcudadevrt)
