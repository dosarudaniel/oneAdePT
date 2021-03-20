cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(porting-cuda-to-sycl LANGUAGES CXX CUDA)

if (NOT SYCL_ROOT)
   set(SYCL_ROOT $ENV{SYCL_ROOT})
endif()

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "SYCL_ROOT not specified! Please specify -DSYCL_ROOT=<DPCPP_ROOT_PATH>")
endif()

set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures")


set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/include/sycl")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS
  	"-fsycl"
  	"-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
  	"-fsycl-unnamed-lambda"
  	"-std=c++20"
  	"-stdlib=libstdc++"
        "-Wno-unknown-cuda-version")
      
set(EXTRA_FLAGS "-DNDEBUG" "-DROOT_NO_INT128" "-I${CMAKE_SOURCE_DIR}/../local/include" )

#include_directories("-I${CMAKE_SOURCE_DIR}/base/inc")

if (EXISTS ${CMAKE_SOURCE_DIR}/extra)
  include(${CMAKE_SOURCE_DIR}/extra/extra.cmake)
endif()

#Find VecCore with correct backend
set(VecCore_VERSION 0.5.2)
set(VecCore_BACKEND CUDA)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_BACKEND})
message(STATUS "Using VecCore version ${VecCore_VERSION}")

# Find VecGeom geometry headers library
set(VecGeom_VERSION 1.1.11)
find_package(VecGeom ${VecGeom_VERSION} REQUIRED)
message(STATUS "Using VecGeom version ${VecGeom_VERSION}")
# make sure we import VecGeom architecture flags - is this needed?
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_CXX_FLAGS}")

# Find Geant4, optional for now
find_package(Geant4 QUIET)
if(Geant4_FOUND)
  message(STATUS "Using Geant4 version ${Geant4_VERSION} from ${Geant4_INCLUDE_DIRS}")
else()
  message(STATUS "Did not find Geant4")
endif()

# Add external dependencies before our own code to allow checking for the
# targets and depend on them.
add_subdirectory(external)

add_subdirectory(examples)

add_subdirectory(tests)

