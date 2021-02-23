// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_atomic.cu
 * @brief Unit test for atomic operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <CL/sycl.hpp>
#include <iostream>

#include <AdePT/Atomic.h>
 
class CUDADeviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    return 1;
  };
};
  
// Example data structure containing several atomics
struct SomeStruct {
  adept::Atomic_t<int> var_int;
  //adept::Atomic_t<float> var_float;
  //sycl::atomic<int, global_space> var_int;
  //sycl::atomic<float, global_space> var_float;

   SomeStruct() {}

  static SomeStruct *MakeInstanceAt(void *addr)
  {
    SomeStruct *obj = new (addr) SomeStruct();
    return obj;
  }
};

// Kernel function to perform atomic addition
void testAdd(SomeStruct *s)
{
  // Test fetch_add, fetch_sub
  s->var_int.fetch_add(1);
  //s->var_float.fetch_add(1);
}

//______________________________________________________________________________________
int main(void)
{
  const sycl::device  &dev_ct1 = sycl::device();

  sycl::queue q_ct1{CUDADeviceSelector()};

  // Allocate the content of SomeStruct in a buffer
  char *buffer = nullptr;
  buffer        = (char *)sycl::malloc_shared(sizeof(SomeStruct), q_ct1);
  SomeStruct *a = SomeStruct::MakeInstanceAt(buffer);
  // Wait memory to reach device
  q_ct1.wait_and_throw();
  
  // Define the kernels granularity: 10K blocks of 32 treads each
  sycl::range<3> nblocks(1, 1, 10000), nthreads(1, 1, 32);
 
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
      testAdd(a);
    });
  });

  q_ct1.wait_and_throw();

  return(0);
}
