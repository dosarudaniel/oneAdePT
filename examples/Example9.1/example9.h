// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE9_H
#define EXAMPLE9_H

#include "Cuda.h"

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmElectronManager.hh>
#include <G4HepEmGammaManager.hh>

void example9(const vecgeom::VPlacedVolume *world, int numParticles, double energy, 
                struct G4HepEmElectronManager *electronManager_p, 
                struct G4HepEmGammaManager *gammaManager_p, 
                struct G4HepEmParameters *g4HepEmPars_p,
                struct G4HepEmData *g4HepEmData_p);

// Interface between C++ and CUDA.

#endif
