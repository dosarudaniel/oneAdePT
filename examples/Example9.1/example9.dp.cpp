// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "example9.h"
#include "example9.dp.hpp"

#include <AdePT/1/Atomic.h>
#include <AdePT/1/LoopNavigator.h>
#include <AdePT/1/MParray.h>

#include <CopCore/1/Global.h>
#include <CopCore/1/PhysicalConstants.h>
#include <CopCore/1/Ranluxpp.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <G4HepEmElectronManager.hh>

#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <chrono>

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in example9.cu)

extern SYCL_EXTERNAL void CopyG4HepEmDataToGPU(struct G4HepEmData* onCPU); 

struct G4HepEmState {
  G4HepEmData data;
  G4HepEmParameters parameters;
};

static G4HepEmState *InitG4HepEm(sycl::queue q_ct1, 
                                  struct G4HepEmElectronManager *electronManager_p, 
                                  struct G4HepEmParameters *g4HepEmPars_p,
                                  struct G4HepEmData *g4HepEmData_p)
{

  electronManager_p =  electronManager.get_ptr();
  
  g4HepEmPars_p =  g4HepEmPars.get_ptr();
  g4HepEmData_p = g4HepEmData.get_ptr();

  G4HepEmState *state = new G4HepEmState;
  InitG4HepEmData(&state->data);
  InitHepEmParameters(&state->parameters);

  InitMaterialAndCoupleData(&state->data, &state->parameters);

  InitElectronData(&state->data, &state->parameters, true);
  InitElectronData(&state->data, &state->parameters, false);
  InitGammaData(&state->data, &state->parameters);

  G4HepEmMatCutData *cutData = state->data.fTheMatCutData;
  std::cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData
            << std::endl;

  // Copy to GPU.
  CopyG4HepEmDataToGPU(&state->data);
  /*
  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((q_ct1
                          .memcpy(g4HepEmPars.get_ptr(), &state->parameters,
                                  sizeof(G4HepEmParameters))
                          .wait(),
                      0));

  // Create G4HepEmData with the device pointers.
  G4HepEmData dataOnDevice;
#ifdef __SYCL_DEVICE_ONLY__
  dataOnDevice.fTheMatCutData   = state->data.fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->data.fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->data.fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->data.fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->data.fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->data.fTheSBTableData_gpu;
  dataOnDevice.fTheGammaData    = state->data.fTheGammaData_gpu;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData_gpu   = nullptr;
  dataOnDevice.fTheMaterialData_gpu = nullptr;
  dataOnDevice.fTheElementData_gpu  = nullptr;
  dataOnDevice.fTheElectronData_gpu = nullptr;
  dataOnDevice.fThePositronData_gpu = nullptr;
  dataOnDevice.fTheSBTableData_gpu  = nullptr;
  dataOnDevice.fTheGammaData_gpu    = nullptr;
#else
  dataOnDevice.fTheMatCutData   = state->data.fTheMatCutData;
  dataOnDevice.fTheMaterialData = state->data.fTheMaterialData;
  dataOnDevice.fTheElementData  = state->data.fTheElementData;
  dataOnDevice.fTheElectronData = state->data.fTheElectronData;
  dataOnDevice.fThePositronData = state->data.fThePositronData;
  dataOnDevice.fTheSBTableData  = state->data.fTheSBTableData;
  dataOnDevice.fTheGammaData    = state->data.fTheGammaData;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData   = nullptr;
  dataOnDevice.fTheMaterialData = nullptr;
  dataOnDevice.fTheElementData  = nullptr;
  dataOnDevice.fTheElectronData = nullptr;
  dataOnDevice.fThePositronData = nullptr;
  dataOnDevice.fTheSBTableData  = nullptr;
  dataOnDevice.fTheGammaData   = nullptr;
 #endif
  /*
  DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK(
      (q_ct1.memcpy(g4HepEmData.get_ptr(), &dataOnDevice, sizeof(G4HepEmData))
           .wait(),
       0));

  return state;
}

static void FreeG4HepEm(G4HepEmState *state)
{
  FreeG4HepEmData(&state->data);
  delete state;
}

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
//  * One for all particles that need to be relocated to the next volume.
struct ParticleQueues {
  adept::MParray *currentlyActive;
  adept::MParray *nextActive;
  adept::MParray *relocate;

  void SwapActive() { std::swap(currentlyActive, nextActive); }
};

struct ParticleType {
  Track *tracks;
  SlotManager *slotManager;
  ParticleQueues queues;
  sycl::queue *stream;
  sycl::event event;
  std::chrono::time_point<std::chrono::steady_clock> event_ct1;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  ParticleQueues queues[ParticleType::NumParticleTypes];
};

// Kernel to initialize the set of queues per particle type.
void InitParticleQueues(ParticleQueues queues, size_t Capacity)
{
  adept::MParray::MakeInstanceAt(Capacity, queues.currentlyActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.nextActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.relocate);
}

// Kernel function to initialize a set of primary particles.
SYCL_EXTERNAL void InitPrimaries(ParticleGenerator generator, int particles, double energy,
		   const vecgeom::VPlacedVolume *world,
                              sycl::nd_item<3> item_ct1)
{
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
       i < particles;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    Track &track = generator.NextTrack();

    track.rngState.SetSeed(314159265 * (i + 1));
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.pos = {0, 0, 0};
    track.dir = {1.0, 0, 0};
    LoopNavigator::LocatePointIn(world, track.pos, track.currentState, true);
    // nextState is initialized as needed.
  }
}

// A data structure to transfer statistics after each iteration.
struct Stats {
  GlobalScoring scoring;
  int inFlight[ParticleType::NumParticleTypes];
};

// Finish iteration: clear queues and fill statistics.
void FinishIteration(AllParticleQueues all, const GlobalScoring *scoring, Stats *stats)
{
  stats->scoring = *scoring;
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    stats->inFlight[i] = all.queues[i].nextActive->size();
    all.queues[i].relocate->clear();
  }
}

void example9(const vecgeom::VPlacedVolume *world, int numParticles, double energy, 
              struct G4HepEmElectronManager *electronManager_p,
              struct G4HepEmParameters *g4HepEmPars_p,
              struct G4HepEmData *g4HepEmData_p)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();

#ifdef __SYCL_DEVICE_ONLY__
  auto &cudaManager = vecgeom::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();
  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();
#else
  const vecgeom::VPlacedVolume *world_dev = world;
#endif
  
  G4HepEmState *state = InitG4HepEm(q_ct1, electronManager_p, g4HepEmPars_p, g4HepEmData_p);

  // Capacity of the different containers aka the maximum number of particles.
  constexpr int Capacity = 256 * 1024;

  std::cout << "INFO: capacity of containers set to " << Capacity << std::endl;

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  constexpr size_t TracksSize  = sizeof(Track) * Capacity;
  constexpr size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize       = adept::MParray::SizeOfInstance(Capacity);

  ParticleType particles[ParticleType::NumParticleTypes];
  SlotManager slotManagerInit(Capacity);
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (particles[i].tracks = (Track *)sycl::malloc_device(TracksSize, q_ct1),
         0));

    /*
    DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (particles[i].slotManager =
             (SlotManager *)sycl::malloc_device(ManagerSize, q_ct1),
         0));
    /*
    DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (q_ct1.memcpy(particles[i].slotManager, &slotManagerInit, ManagerSize)
             .wait(),
         0));

    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (particles[i].queues.currentlyActive =
             (adept::MParray *)sycl::malloc_device(QueueSize, q_ct1),
         0));
    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (particles[i].queues.nextActive =
             (adept::MParray *)sycl::malloc_device(QueueSize, q_ct1),
         0));
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (particles[i].queues.relocate =
             (adept::MParray *)sycl::malloc_device(QueueSize, q_ct1),
         0));
    q_ct1.submit([&](sycl::handler &cgh) {
      auto particles_i_queues_ct0 = particles[i].queues;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            InitParticleQueues(particles_i_queues_ct0, Capacity);
          });
    });

    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((particles[i].stream = dev_ct1.create_queue(), 0));
    /*
    DPCT1027:9: The call to cudaEventCreate was replaced with 0, because this
    call is redundant in DPC++.
    */
    COPCORE_CUDA_CHECK(0);
  }
  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));

  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  // Create a stream to synchronize kernels of all particle types.
  sycl::queue *stream;
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((stream = dev_ct1.create_queue(), 0));

  // Allocate and initialize scoring and statistics.
  GlobalScoring *scoring = nullptr;
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK(
      (scoring = sycl::malloc_device<GlobalScoring>(1, q_ct1), 0));
  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK(
      (q_ct1.memset(scoring, 0, sizeof(GlobalScoring)).wait(), 0));

  Stats *stats_dev = nullptr;
  /*
  DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((stats_dev = sycl::malloc_device<Stats>(1, q_ct1), 0));
  Stats *stats = nullptr;
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((stats = sycl::malloc_host<Stats>(1, q_ct1), 0));

  // Initialize primary particles.
  constexpr int InitThreads = 32;
  int initBlocks            = (numParticles + InitThreads - 1) / InitThreads;
  ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, initBlocks) *
                                       sycl::range<3>(1, 1, InitThreads),
                                       sycl::range<3>(1, 1, InitThreads)),
                     [=](sycl::nd_item<3> item_ct1) {
                       InitPrimaries(electronGenerator, numParticles, energy,
                                     world_dev, item_ct1);
                     });
  });
  /*
  DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));

  stats->inFlight[ParticleType::Electron] = numParticles;
  stats->inFlight[ParticleType::Positron] = 0;
  stats->inFlight[ParticleType::Gamma]    = 0;

  std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T";
  std::cout << std::endl;

  constexpr int MaxBlocks        = 1024;
  constexpr int TransportThreads = 32;
  constexpr int RelocateThreads  = 32;
  int transportBlocks, relocateBlocks;

  vecgeom::Stopwatch timer;
  timer.Start();

  int inFlight;
  int iterNo = 0;

  do {
    Secondaries secondaries = {
        .electrons = {electrons.tracks, electrons.slotManager, electrons.queues.nextActive},
        .positrons = {positrons.tracks, positrons.slotManager, positrons.queues.nextActive},
        .gammas    = {gammas.tracks, gammas.slotManager, gammas.queues.nextActive},
    };

    // *** ELECTRONS ***
    int numElectrons = stats->inFlight[ParticleType::Electron];
    if (numElectrons > 0) {
      transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      relocateBlocks = std::min(numElectrons, MaxBlocks);

      electrons.stream->submit([&](sycl::handler &cgh) {
        Track *electronsTracks = electrons.tracks;
        adept::MParray *currentlyActive = electrons.queues.currentlyActive;
        adept::MParray *nextActive = electrons.queues.nextActive;
        adept::MParray *relocate = electrons.queues.relocate;
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, transportBlocks) *
                                  sycl::range<3>(1, 1, TransportThreads),
                              sycl::range<3>(1, 1, TransportThreads)),
            [=](sycl::nd_item<3> item_ct1) {
              TransportElectrons<true>(electronsTracks,
                                       currentlyActive,
                                       secondaries, 
                                       nextActive,
                                       relocate, 
                                       scoring, 
                                       item_ct1,
                                       electronManager_p,
                                       g4HepEmPars_p,
                                       g4HepEmData_p);
            });
      });
      /*
      RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, electrons.stream>>>(electrons.tracks,
                                                                                     electrons.queues.relocate);
      */
      /*
      DPCT1012:17: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:18: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      electrons.event_ct1 = std::chrono::steady_clock::now();
      COPCORE_CUDA_CHECK(0);
      /*
      DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      COPCORE_CUDA_CHECK((electrons.event.wait(), 0));
    }

    // *** POSITRONS ***
    int numPositrons = stats->inFlight[ParticleType::Positron];
    if (numPositrons > 0) {
      transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      relocateBlocks = std::min(numPositrons, MaxBlocks);

      positrons.stream->submit([&](sycl::handler &cgh) {
        Track *positronsTracks = positrons.tracks;
        adept::MParray *pCurrentlyActive = positrons.queues.currentlyActive;
        adept::MParray *pNextActive = positrons.queues.nextActive;
        adept::MParray *pRelocate = positrons.queues.relocate;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, transportBlocks) *
                                  sycl::range<3>(1, 1, TransportThreads),
                              sycl::range<3>(1, 1, TransportThreads)),
            [=](sycl::nd_item<3> item_ct1) {
              TransportElectrons<false>(positronsTracks,
                                        pCurrentlyActive,
                                        secondaries,
                                        pNextActive,
                                        pRelocate,
                                        scoring,
                                        item_ct1,
                                        electronManager_p,
                                        g4HepEmPars_p,
                                        g4HepEmData_p);
	    });
      });
      /*
      RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, positrons.stream>>>(positrons.tracks,
                                                                                     positrons.queues.relocate);
      */
      /*
      DPCT1012:20: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:21: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      positrons.event_ct1 = std::chrono::steady_clock::now();
      COPCORE_CUDA_CHECK(0);
      /*
      DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      COPCORE_CUDA_CHECK((positrons.event.wait(), 0));
    }

    // *** GAMMAS ***
    int numGammas = stats->inFlight[ParticleType::Gamma];
    if (numGammas > 0) {
      transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      relocateBlocks = std::min(numGammas, MaxBlocks);

      gammas.stream->submit([&](sycl::handler &cgh) {
        Track *gammasTracks = gammas.tracks;
        adept::MParray *gCurrentlyActive = gammas.queues.currentlyActive;
        adept::MParray *gNextActive = gammas.queues.nextActive;
        adept::MParray *gRelocate = gammas.queues.relocate;
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, transportBlocks) *
                                  sycl::range<3>(1, 1, TransportThreads),
                              sycl::range<3>(1, 1, TransportThreads)),
            [=](sycl::nd_item<3> item_ct1) {
              TransportGammas(gammasTracks,
                              gCurrentlyActive,
                              secondaries,
                              gNextActive,
                              gRelocate,
                              scoring,
                              item_ct1,
                              electronManager_p,
                              g4HepEmPars_p,
                              g4HepEmData_p);
            });
      });
      /*
      RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, gammas.stream>>>(gammas.tracks,
                                                                                  gammas.queues.relocate);
      */
      /*
      DPCT1012:23: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:24: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      gammas.event_ct1 = std::chrono::steady_clock::now();
      COPCORE_CUDA_CHECK(0);
      /*
      DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      COPCORE_CUDA_CHECK((gammas.event.wait(), 0));
    }

    // *** END OF TRANSPORT ***

    // The events ensure synchronization before finishing this iteration and
    // copying the Stats back to the host.
    AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
    stream->submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            FinishIteration(queues, scoring, stats_dev);
          });
    });
    /*
    DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((stream->memcpy(stats, stats_dev, sizeof(Stats)), 0));

    // Finally synchronize all kernels.
    /*
    DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((stream->wait(), 0));

    // Count the number of particles in flight.
    inFlight = 0;
    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      inFlight += stats->inFlight[i];
    }

    // Swap the queues for the next iteration.
    electrons.queues.SwapActive();
    positrons.queues.SwapActive();
    gammas.queues.SwapActive();

    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    std::cout << "iter " << std::setw(4) << iterNo << " -- tracks in flight: " << std::setw(5) << inFlight
              << " energy deposition: " << std::setw(10) << stats->scoring.energyDeposit / copcore::units::GeV
              << " number of secondaries: " << std::setw(5) << stats->scoring.secondaries
              << " number of hits: " << std::setw(4) << stats->scoring.hits;
    std::cout << std::endl;

    iterNo++;
  } while (inFlight > 0 && iterNo < 1000);

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";

  // Free resources.
  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(scoring, q_ct1), 0));
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(stats_dev, q_ct1), 0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(stats, q_ct1), 0));

  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.destroy_queue(stream), 0));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    /*
    DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].tracks, q_ct1), 0));
    /*
    DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].slotManager, q_ct1), 0));

    /*
    DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (sycl::free(particles[i].queues.currentlyActive, q_ct1), 0));
    /*
    DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].queues.nextActive, q_ct1), 0));
    /*
    DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].queues.relocate, q_ct1), 0));

    /*
    DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((dev_ct1.destroy_queue(particles[i].stream), 0));
    /*
    DPCT1027:38: The call to cudaEventDestroy was replaced with 0, because this
    call is redundant in DPC++.
    */
    COPCORE_CUDA_CHECK(0);
  }

  FreeG4HepEm(state);
}
