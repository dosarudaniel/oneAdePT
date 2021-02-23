// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <iostream>

#include <AdePT/BlockData.h>

class deviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    return 1;
  }
};

using Queue_t = adept::mpmc_bounded_queue<int>;

void select_process(Queue_t *queues[], sycl::nd_item<3> item_ct1)
{
  //queues[0]->enqueue(0);
}


int main()
{
  //  const sycl::device  &dev_ct1 = sycl::device();

  sycl::queue q_ct1{deviceSelector()};
  
  using Queue_t = adept::mpmc_bounded_queue<int>;

  Queue_t **queues = nullptr;
  queues           = sycl::malloc_shared<Queue_t *>(3, q_ct1);

  sycl::range<3> nthreads(1, 1, 32);

  q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
          select_process(queues, item_ct1);
      });
  });
  return 0;
}
