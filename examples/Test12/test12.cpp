#include <CL/sycl.hpp>
#include <AdePT/1/MParray.h>

extern SYCL_EXTERNAL double stepInField(double kinE, double mass, int charge); // Check field.cu file

void kernel(double *step)
{
  #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
     *step = stepInField(2.0, 3.0, 1);
  #endif
}

void kernel2()
{

}



// struct Track {
//   RanluxppDouble rngState;
//   double energy;
//   double numIALeft[3];

//   vecgeom::Vector3D<double> pos;
//   vecgeom::Vector3D<double> dir;
//   vecgeom::NavStateIndex currentState;
//   vecgeom::NavStateIndex nextState;

//   double Uniform() { return rngState.Rndm(); }

//   void SwapStates()
//   {
//     auto state         = this->currentState;
//     this->currentState = this->nextState;
//     this->nextState    = state;
//   }

//   void InitAsSecondary(const Track &parent)
//   {
//     // Initialize a new PRNG state.
//     this->rngState = parent.rngState;
//     this->rngState.Skip(1 << 15);

//     // The caller is responsible to set the energy.
//     this->numIALeft[0] = -1.0;
//     this->numIALeft[1] = -1.0;
//     this->numIALeft[2] = -1.0;

//     // A secondary inherits the position of its parent; the caller is responsible
//     // to update the directions.
//     this->pos          = parent.pos;
//     this->currentState = parent.currentState;
//     this->nextState    = parent.nextState;
//   }
// };

// // A data structure to manage slots in the track storage.
// class SlotManager {
//   adept::Atomic_t<int> fNextSlot;
//   const int fMaxSlot;

// public:
//   SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

//   int NextSlot()
//   {
//     int next = fNextSlot.fetch_add(1);
//     if (next >= fMaxSlot) return -1;
//     return next;
//   }
// };

struct ParticleQueues {
  adept::MParray *currentlyActive;
};

struct ParticleType {
  // Track *tracks;
  // SlotManager *slotManager;
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

int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
        	<< q_ct1.get_device().get_info<cl::sycl::info::device::name>()
        	<< "\n";
 
  double *d_dev_ptr;

  d_dev_ptr  = sycl::malloc_shared<double>(1, q_ct1);

  ParticleType particles;

  constexpr int Capacity = 256 * 1024;
  const size_t QueueSize       = adept::MParray::SizeOfInstance(Capacity);

  particles.queues.currentlyActive = (adept::MParray *)sycl::malloc_device(QueueSize, q_ct1);


  q_ct1.submit([&](sycl::handler& cgh) {
    auto particles_i_queues_ct0 = particles.queues;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {

            adept::MParray::MakeInstanceAt(Capacity, particles_i_queues_ct0.currentlyActive);

    });
  });



  double d_dev;
  q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();
  std::cout << "   device: " << d_dev << std::endl;
  sycl::free(d_dev_ptr, q_ct1);
}

