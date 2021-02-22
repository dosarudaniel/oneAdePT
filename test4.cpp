// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <AdePT/BlockData.h>

class deviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    return 1;
  }
};

int main()
{
  sycl::queue q_ct1{deviceSelector()};

  adept::mpmc_bounded_queue<int> **queues = 
      sycl::malloc_shared<adept::mpmc_bounded_queue<int> *>(1, q_ct1);

  sycl::range<3> nthreads(1, 1, 1);

  q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
          queues[0]->enqueue(0);
      });
  });
  return 0;
}
