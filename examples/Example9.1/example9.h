// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE9_H
#define EXAMPLE9_H

#include "Cuda.h"

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

void example9(const vecgeom::VPlacedVolume *world, int numParticles, double energy);

// Interface between C++ and CUDA.

#endif
