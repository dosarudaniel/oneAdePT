// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <iostream>
#include <CopCore/1/Ranluxpp.h>
#include <AdePT/1/BlockData.h>

using Queue_t = adept::mpmc_bounded_queue<int>;

// kernel function that does transportation

void test(Queue_t *queues, RanluxppDouble *r, double *ret, sycl::nd_item<1> item_ct1)
{
  double p = r->Rndm();

  if ( p > 0.5f) {
    queues[0].enqueue(0);
  } else {
    queues[1].enqueue(0);
  }

  *ret = p;
}

void kernel(RanluxppDouble *r, double *d)
{
  float p = r->Rndm();
  if ( p < 0.5 ) {
    *d = p + 1;
  } else {
    *d = p + 2;
  }
}

int main()
{ 
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

   
  RanluxppDouble r;
  
  //std::cout << "double: " << r.Rndm() << std::endl;
 
  RanluxppDouble *r_dev = sycl::malloc_shared<RanluxppDouble>(1, q_ct1);

  double *d_dev_ptr  = sycl::malloc_shared<double>(1, q_ct1);

  // Transfer the state of the generator to the device.
  q_ct1.memcpy(r_dev, &r, sizeof(RanluxppDouble)).wait();

  //q_ct1.wait_and_throw();
  
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       kernel(r_dev, d_dev_ptr);
                     });
  });

  // Generate from the same state on the host.
  q_ct1.wait_and_throw();

  double d   = r.Rndm();

  std::cout << std::endl;
  std::cout << "   host:   " << d << std::endl;
  std::cout << "   device: " << *d_dev_ptr << std::endl;
  
  // Capacity of the different containers
  constexpr int capacity = 1 << 20;

  using Queue_t = adept::mpmc_bounded_queue<int>;

  constexpr int numberOfProcesses = 3;

  Queue_t **queues  = sycl::malloc_shared<Queue_t *>(numberOfProcesses, q_ct1);

  size_t buffersize = Queue_t::SizeOfInstance(capacity);

  char *buffer;
  for (int i = 0; i < numberOfProcesses; i++) {
    buffer = (char *)sycl::malloc_shared(buffersize, q_ct1);
    queues[i] = Queue_t::MakeInstanceAt(capacity, buffer);
  }
  
  int numBlocks;
  int nthreads = 32;
     
   numBlocks = 1;

   q_ct1.submit([&](sycl::handler &cgh) {
	      cgh.parallel_for(sycl::nd_range<1>(numBlocks * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
		    test(queues[2], r_dev, d_dev_ptr, item_ct1);
         });
   });

   q_ct1.wait_and_throw();

   for (int i = 0; i < numberOfProcesses; i++) {
     std::cout << i << " select " << queues[i]->size() << "\n";
   }

  //q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();

  //q_ct1.wait_and_throw();

  // std::cout << "p=" << d_dev << "\n";  

 }
