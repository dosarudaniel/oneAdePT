#include <CL/sycl.hpp>
// #include <AdePT/1/MParray.h>

// extern SYCL_EXTERNAL double stepInField(double kinE, double mass, int charge); // Check field.cu file

// void kernel(double *step)
// {
//   #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
//      *step = stepInField(2.0, 3.0, 1);
//   #endif
// }

void kernel2()
{

}

// struct ParticleQueues {
//   adept::MParray *currentlyActive;
// };

// struct ParticleType {
//   // Track *tracks;
//   // SlotManager *slotManager;
//   ParticleQueues queues;
//   sycl::queue *stream;
//   sycl::event event;
//   std::chrono::time_point<std::chrono::steady_clock> event_ct1;

//   enum {
//     Electron = 0,
//     Positron = 1,
//     Gamma    = 2,

//     NumParticleTypes,
//   };
// };


int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
                                        sycl::range<3>(1, 1, 1)), 
                    [=](sycl::nd_item<3> item_ct1) {
      kernel2();
    });
  }).wait();
}

