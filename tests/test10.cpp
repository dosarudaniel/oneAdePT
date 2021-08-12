#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>

#include <Field/1/fieldPropagatorConstBz.h>
#include <CopCore/1/SystemOfUnits.h>
#include <CopCore/1/PhysicalConstants.h>

#include <AdePT/1/MParray.h>

// A data structure to manage slots in the track storage.
class SlotManager {
  adept::Atomic_t<int> fNextSlot;
  const int fMaxSlot;

public:
  SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

  int NextSlot()
  {
    int next = fNextSlot.fetch_add(1);
    if (next >= fMaxSlot) return -1;
    return next;
  }
};


struct Track {
  //RanluxppDouble rngState;
  double energy;
  double numIALeft[3];

  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  vecgeom::NavStateIndex currentState;
  vecgeom::NavStateIndex nextState;

  //double Uniform() { return rngState.Rndm(); }

  void SwapStates()
  {
    auto state         = this->currentState;
    this->currentState = this->nextState;
    this->nextState    = state;
  }

  // void InitAsSecondary(const Track &parent)
  // {
  //   // Initialize a new PRNG state.
  //   this->rngState = parent.rngState;
  //   this->rngState.Skip(1 << 15);

  //   // The caller is responsible to set the energy.
  //   this->numIALeft[0] = -1.0;
  //   this->numIALeft[1] = -1.0;
  //   this->numIALeft[2] = -1.0;

  //   // A secondary inherits the position of its parent; the caller is responsible
  //   // to update the directions.
  //   this->pos           = parent.pos;
  //   this->currentState = parent.currentState;
  //   this->nextState    = parent.nextState;
  // }
};

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


constexpr float BzFieldValue = 0.1 * copcore::units::tesla;
constexpr double Mass = copcore::units::kElectronMassC2;
constexpr int Charge  = -1; // IsElectron ? -1 : 1;

void kernel(double *d, Track *electrons)
{
  *d = sycl::log(*d);

  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);
  double geometricalStepLengthFromPhysics = 0.0; // theTrack->GetGStepLength();
  Track &currentTrack = electrons[0]; // Track &currentTrack = electrons[slot];
  fieldPropagatorBz.ComputeStepAndPropagatedState<false>(
                                                    0.0 /*dumb energy*/,
                                                    Mass,
                                                    Charge,
                                                    geometricalStepLengthFromPhysics,
                                                    currentTrack.pos,
                                                    currentTrack.dir,
                                                    currentTrack.currentState,
                                                    currentTrack.nextState);
}


int main(void)
{


  ParticleType particles[ParticleType::NumParticleTypes];
  ParticleType &electrons = particles[ParticleType::Electron];
  Track *electronsTracks = electrons.tracks;

  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  double arg = 10;
  
  double *d_dev_ptr;
  d_dev_ptr  = sycl::malloc_device<double>(1, q_ct1);
  q_ct1.memcpy(d_dev_ptr, &arg, sizeof(double)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       kernel(d_dev_ptr, electronsTracks);
                     });
  }).wait();
 
  double d_dev;
  q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();

  std::cout << "   host:   " << std::log(arg) << std::endl;
  std::cout << "   device: " << d_dev << std::endl;

  sycl::free(d_dev_ptr, q_ct1);

}
