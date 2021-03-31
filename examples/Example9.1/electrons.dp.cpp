// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "example9.dp.hpp"

#include <Field/1/fieldPropagatorConstBz.h>

#include <CopCore/1/PhysicalConstants.h>

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>

struct G4HepEmElectronManager electronManager;

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron>
void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                        adept::MParray *activeQueue , adept::MParray *relocateQueue, GlobalScoring *scoring,
			                  sycl::nd_item<3> item_ct1)
{
  constexpr int Charge  = IsElectron ? -1 : 1;
  constexpr double Mass = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = active->size();
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
       i < activeSize;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    auto volume         = currentTrack.currentState.Top();
    if (volume == nullptr) {
      // The particle left the world, kill it by not enqueuing into activeQueue.
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    // For now, just assume a single material.
    int theMCIndex = 1;
    theTrack->SetMCIndex(theMCIndex);
    theTrack->SetCharge(Charge);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -sycl::log(currentTrack.Uniform());
        currentTrack.numIALeft[ip] = numIALeft;
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    electronManager.HowFar(&g4HepEmData, &g4HepEmPars, &elTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    double geometryStepLength = fieldPropagatorBz.ComputeStepAndPropagatedState</*Relocate=*/false>(
        currentTrack.energy, Mass, Charge, geometricalStepLengthFromPhysics, currentTrack.pos, currentTrack.dir,
        currentTrack.currentState, currentTrack.nextState);

    if (currentTrack.nextState.IsOnBoundary()) {
      theTrack->SetGStepLength(geometryStepLength);
      theTrack->SetOnBoundary(true);
    }

    // Apply continuous effects.
    bool stopped = electronManager.PerformContinuous(&g4HepEmData,
                                                      &g4HepEmPars, &elTrack);
    // Collect the changes.
    currentTrack.energy = theTrack->GetEKin();
    dpct::atomic_fetch_add(&scoring->energyDeposit,
                           theTrack->GetEnergyDeposit());

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (stopped) {
      if (!IsElectron) {
        // Annihilate the stopped positron into two gammas heading to opposite
        // directions (isotropic).
        Track &gamma1 = secondaries.gammas.NextTrack();
        Track &gamma2 = secondaries.gammas.NextTrack();
        sycl::atomic<int>(sycl::global_ptr<int>(&scoring->secondaries))
            .fetch_add(2);

        const double cost = 2 * currentTrack.Uniform() - 1;
        const double sint = sycl::sqrt(1 - cost * cost);
        const double phi  = k2Pi * currentTrack.Uniform();
        double sinPhi, cosPhi;
        /*
        DPCT1017:0: The sycl::sincos call is used instead of the sincos call.
        These two calls do not provide exactly the same functionality. Check the
        potential precision and/or performance issues for the generated code.
        */
        sinPhi = sycl::sincos(
            phi,
            sycl::make_ptr<double, sycl::access::address_space::global_space>(
                &cosPhi));

        gamma1.InitAsSecondary(/*parent=*/currentTrack);
        gamma1.energy = copcore::units::kElectronMassC2;
        gamma1.dir.Set(sint * cosPhi, sint * sinPhi, cost);

        gamma2.InitAsSecondary(/*parent=*/currentTrack);
        gamma2.energy = copcore::units::kElectronMassC2;
        gamma2.dir    = -gamma1.dir;
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      continue;
    }

    if (currentTrack.nextState.IsOnBoundary()) {
      // For now, just count that we hit something.
      sycl::atomic<int>(sycl::global_ptr<int>(&scoring->hits)).fetch_add(1);

      activeQueue->push_back(slot);
      // relocateQueue->push_back(slot);

      //LoopNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);

      // Move to the next boundary.
      currentTrack.SwapStates();
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (electronManager.CheckDelta(&g4HepEmData, theTrack,
                                    currentTrack.Uniform())) {
      // A delta interaction happened, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Perform the discrete interaction.
    RanluxppDoubleEngine rnge(&currentTrack.rngState);

    const double energy   = currentTrack.energy;
    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fSecElProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin = (IsElectron) ? SampleETransferMoller(theElCut, energy, &rnge)
                                      : SampleETransferBhabha(theElCut, energy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      SampleDirectionsIoni(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &secondary = secondaries.electrons.NextTrack();
      sycl::atomic<int>(sycl::global_ptr<int>(&scoring->secondaries))
          .fetch_add(1);

      secondary.InitAsSecondary(/*parent=*/currentTrack);
      secondary.energy = deltaEkin;
      secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.energy = energy - deltaEkin;
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      // The current track continues to live.
      activeQueue->push_back(slot);
      break;
    }
    case 1: {
      // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
      double logEnergy = sycl::log((double)energy);
      double deltaEkin = energy < g4HepEmPars.fElectronBremModelLim
                             ? SampleETransferBremSB(&g4HepEmData, energy, logEnergy, theMCIndex, &rnge, IsElectron)
                             : SampleETransferBremRB(&g4HepEmData, energy, logEnergy, theMCIndex, &rnge, IsElectron);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      SampleDirectionsBrem(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &gamma = secondaries.gammas.NextTrack();
      sycl::atomic<int>(sycl::global_ptr<int>(&scoring->secondaries))
          .fetch_add(1);

      gamma.InitAsSecondary(/*parent=*/currentTrack);
      gamma.energy = deltaEkin;
      gamma.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.energy = energy - deltaEkin;
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      // The current track continues to live.
      activeQueue->push_back(slot);
      break;
    }
    case 2: {
      // Invoke annihilation (in-flight) for e+
      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double theGamma1Ekin, theGamma2Ekin;
      double theGamma1Dir[3], theGamma2Dir[3];
      SampleEnergyAndDirectionsForAnnihilationInFlight(energy, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin,
                                                       theGamma2Dir, &rnge);

      Track &gamma1 = secondaries.gammas.NextTrack();
      Track &gamma2 = secondaries.gammas.NextTrack();
      sycl::atomic<int>(sycl::global_ptr<int>(&scoring->secondaries))
          .fetch_add(2);

      gamma1.InitAsSecondary(/*parent=*/currentTrack);
      gamma1.energy = theGamma1Ekin;
      gamma1.dir.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);

      gamma2.InitAsSecondary(/*parent=*/currentTrack);
      gamma2.energy = theGamma2Ekin;
      gamma2.dir.Set(theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}

// Instantiate template for electrons and positrons.
template void TransportElectrons<true>(Track *electrons, const adept::MParray *active,
               Secondaries secondaries, adept::MParray *activeQueue,
				       adept::MParray *relocateQueue, GlobalScoring *scoring,
				       sycl::nd_item<3> item_ct1);

template void TransportElectrons<false>(Track *electrons, const adept::MParray *active,
              Secondaries secondaries, adept::MParray *activeQueue,
              adept::MParray *relocateQueue,GlobalScoring *scoring,
              sycl::nd_item<3> item_ct1);

