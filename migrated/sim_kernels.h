// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <AdePT/BlockData.h>
#include <AdePT/MParray.h>
#include <AdePT/Atomic.h>

/** @brief Data structures */
struct MyTrack {
  int index{0};
  float energy{0};

  MyTrack() {}
  MyTrack(float e) { energy = e; }
};

struct MyHit {
  adept::Atomic_t<float> edep;
};

// portable kernel functions have to reside in a backend-dependent inline namespace to avoid symbol duplication
// when building executables/libraries running same functions on different backends
inline namespace COPCORE_IMPL {

/** @brief Generate a number of primaries.
    @param id The thread id matching blockIdx.x * blockDim.x + threadIdx.x for CUDA
              The current index of the loop over the input data
    @param tracks Pointer to the container of tracks
  */

void generateAndStorePrimary(int id, adept::BlockData<MyTrack> *tracks)
{
  auto track = tracks->NextElement();
  if (!track) COPCORE_EXCEPTION("generateAndStorePrimary: Not enough space for tracks");

  track->index  = id;
  track->energy = 100.;
}

// Mandatory callable function decoration (storage for device function pointer in global variable)
dpct::global_memory<COPCORE_CALLABLE_FUNC, 0> _ptr_generateAndStorePrimary(generateAndStorePrimary);
  bool selected = (track.index % each_n == 0);
  if (selected && !array->push_back(id)) {
    // Array too small - throw an exception
    COPCORE_EXCEPTION("Array too small. Aborting...\n");
  }
}
dpct::global_memory<COPCORE_CALLABLE_FUNC, 0> _ptr_selectTrack(selectTrack);
  auto &track  = (*tracks)[track_id];
  auto &hit    = (*hits)[id % 1024];
  float edep   = 0.1 * track.energy;
  hit.edep += edep;
  track.energy -= edep;
}
COPCORE_CALLABLE_FUNC(elossTrack)

} // End namespace COPCORE_IMPL
