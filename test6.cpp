// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_atomic.cu
 * @brief Unit test for atomic operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <CL/sycl.hpp>
#include <iostream>
#include <cassert>
#include <AdePT/1/Atomic.h>

struct SomeStruct {
  int a = 0;
  float b = 0.0;
  adept::Atomic_t<int> var_int;
  adept::Atomic_t<float> var_float;

  SomeStruct() : var_int(a), var_float(b) {}

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
  s->var_float.fetch_add(1);
}

// Kernel function to perform atomic subtraction
void testSub(SomeStruct *s)
{
  // Test fetch_add, fetch_sub
  s->var_int.fetch_sub(1);
  s->var_float.fetch_sub(1);
}

// Kernel function to test compare_exchange
void testCompareExchange(SomeStruct *s)
{
  // Read the content of the atomic

  auto expected = s->var_int.load();
  bool success  = false;

  while (!success) {
    // Try to decrement the content, if zero try to replace it with 100
    while (expected > 0) {
      success = s->var_int.compare_exchange_strong(expected, expected - 1);
      if (success) return;
    }
    while (expected == 0) {
      success = s->var_int.compare_exchange_strong(expected, 100);
    }
  }

}

///______________________________________________________________________________________
int main(void)
{
  sycl::default_selector device_selector;
  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  const char *result[2] = {"FAILED", "OK"};
  bool success          = true;
  // Define the kernels granularity: 10K blocks of 32 treads each
  sycl::range<3> nblocks(1, 1, 10000), nthreads(1, 1, 32);

  // Allocate the content of SomeStruct in a buffer
  char *buffer = nullptr;
  buffer        = (char *)sycl::malloc_shared(sizeof(SomeStruct), q_ct1);
  SomeStruct *a = SomeStruct::MakeInstanceAt(buffer);

    // Wait memory to reach device
  q_ct1.wait_and_throw();

  // Launch a kernel doing additions
  bool testOK = true;
  std::cout << "   testAdd ... ";
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
      testAdd(a);
    });
  });
  // Wait all warps to finish and sync memory
  q_ct1.wait_and_throw();

  testOK &= a->var_int.load() == nblocks[2] * nthreads[2];
  testOK &= a->var_float.load() == float(nblocks[2] * nthreads[2]);
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel doing subtraction
  testOK = true;
  std::cout << "   testSub ... ";
  a->var_int.store(nblocks[2] * nthreads[2]);
  a->var_float.store(nblocks[2] * nthreads[2]);
  q_ct1.wait_and_throw();

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
      testSub(a);
    });
  });
  q_ct1.wait_and_throw();

  testOK &= a->var_int.load() == 0;
  testOK &= a->var_float.load() == 0;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel testing compare and swap operations
  std::cout << "   testCAS ... ";
  a->var_int.store(99);
  q_ct1.wait_and_throw();

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
      testCompareExchange(a);
    });
  });
  q_ct1.wait_and_throw();
  testOK = a->var_int.load() == 99;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  sycl::free(buffer, q_ct1);
  if (!success) return 1;
  return 0;
}
