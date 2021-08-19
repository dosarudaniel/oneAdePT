# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example9 (needs G4HepEm)")
  return()
endif()

# Example 9 of particle transportation with GPUs: Electrons and gammas are
# stored in separate containers with monotonic slot numbers, slots are not
# reused. The example includes geometry, magnetic field with constant Bz, and
# physics processes for e-/e+ as well as gammas. Results are reproducible using
# one RANLUX++ state per track.
set(CMAKE_C_COMPILER "${SYCL_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${SYCL_ROOT}/bin/clang++")
set(CMAKE_CXX_STANDARD 20)

add_executable(example9.1 example9.cpp example9.dp.cpp electrons.dp.cpp gammas.dp.cpp) #  relocation.cu)
target_include_directories(example9.1 PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc/G4HepEm>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
      $<INSTALL_INTERFACE:base>)
target_link_libraries(example9.1 PRIVATE VecGeom::vgdml VecGeom::vecgeom VecGeom::vecgeomcuda_static  ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun ${SYCL_FLAGS} ${EXTRA_FLAGS})
target_compile_options(example9.1 PUBLIC ${SYCL_FLAGS} ${EXTRA_FLAGS})

# force the compiler to use the OpenCL from sycl, not the one from CUDA (older version)
target_include_directories(example9.1 SYSTEM BEFORE PUBLIC ${SYCL_INCLUDE_DIR})
set_target_properties(example9.1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

