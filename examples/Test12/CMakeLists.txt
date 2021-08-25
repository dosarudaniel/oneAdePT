# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example9 (needs G4HepEm)")
  return()
endif()






set(CMAKE_C_COMPILER "${SYCL_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${SYCL_ROOT}/bin/clang++")
set(CMAKE_CXX_STANDARD 20)

add_executable(test12 test12.cpp)
target_include_directories(test12 PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc/G4HepEm>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
      $<INSTALL_INTERFACE:base>)
target_link_libraries(test12 PRIVATE VecGeom::vgdml VecGeom::vecgeom VecGeom::vecgeomcuda_static  ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun ${SYCL_FLAGS} ${EXTRA_FLAGS})
# target_link_libraries(test12 PRIVATE ${SYCL_FLAGS} ${EXTRA_FLAGS})

target_compile_options(test12 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})
# target_link_options(test12 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})

# force the compiler to use the OpenCL from sycl, not the one from CUDA (older version)
target_include_directories(test12 SYSTEM BEFORE PUBLIC ${SYCL_INCLUDE_DIR})
# set_target_properties(test12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
