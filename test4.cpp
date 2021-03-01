// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <iostream>

#include <AdePT/1/BlockData.h>

using Queue_t = adept::mpmc_bounded_queue<int>;

void select_process(Queue_t *queues[], sycl::nd_item<3> item_ct1)
{
  queues[0]->enqueue(0);
}

int main()
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  constexpr int capacity = 1 << 20;

  using Queue_t = adept::mpmc_bounded_queue<int>;

  constexpr int numberOfProcesses = 3;

  char *buffer[numberOfProcesses];

  Queue_t **queues = nullptr;
  queues           = sycl::malloc_shared<Queue_t *>(numberOfProcesses, q_ct1);

  size_t buffersize = Queue_t::SizeOfInstance(capacity);

  for (int i = 0; i < numberOfProcesses; i++) {
    std::cout << i ; 
    buffer[i] = (char *)sycl::malloc_shared(buffersize, q_ct1);
    queues[i] = Queue_t::MakeInstanceAt(capacity, buffer[i]);  // It crashes here
  }
  /*
  sycl::range<3> nthreads(1, 1, 32);

  q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
          select_process(queues, item_ct1);
      });
  });
  */
  return 0;
}

 
 
