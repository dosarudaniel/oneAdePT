// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE9_H
#define EXAMPLE9_H

#include "Cuda.h"

/*
#ifdef __SYCL_DEVICE_ONLY__
  #define __constant__
  #define VECGEOM_DEVICE_COMPILATION
  #define VECCORE_CUDA_DEVICE_COMPILATION
  #define VECGEOM_IMPL_NAMESPACE cuda
  #pragma message("__SYCL_DEVICE_ONLY__")
#else
  #pragma message("NOT __SYCL_DEVICE_ONLY__")
#endif
*/

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif
/*
#ifdef VECGEOM_DEVICE_COMPILATION
using namespace vecgeom::cuda;
#else
using namespace vecgeom::cxx;
#endif
*/

void example9(const vecgeom::VPlacedVolume *world, int numParticles, double energy);

// Interface between C++ and CUDA.


#endif
