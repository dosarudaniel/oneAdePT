#include <CL/sycl.hpp>
#include <cstdlib>
#include <cstdio>

#define N 100

void custom_kernel(int * dev_a)
{
  for(int i=0; i<N; i++) {
      dev_a[i] *= 2;
   }
}

int main() {

  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";
  int *dev_a;

  dev_a = sycl::malloc_shared<int>(N, q_ct1);

  for(int i=0; i<N; i++) {
      dev_a[i] = i;
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          custom_kernel(dev_a);
        });
  });

  q_ct1.wait_and_throw();

  for(int i=0; i<N; i++) {
      printf("%d %d\n",i,dev_a[i]);
  }

}
  
