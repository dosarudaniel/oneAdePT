// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_atomic.cu
 * @brief Unit test for atomic operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include <cassert>
#include <AdePT/Atomic.h>


class CUDADeviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    return 1;
    /*
    if (device.get_platform().get_backend() == sycl::backend::cuda)
      return 1;
    else
      return -1;
    */
  }
};

// Example data structure containing several atomics
struct SomeStruct {
  adept::Atomic_t<int> var_int;
  adept::Atomic_t<float> var_float;

  
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
  //  s->var_float.fetch_add(1);
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
  //dpct::device_ext &dev_ct1 = dpct::get_current_device();
  //const sycl::device  &dev_ct1 = sycl::device();
  /*
  std::cout << "Default device type: " << get_type(dev_ct1) << std::endl;

  for (const auto& dev : sycl::device::get_devices()) {
    std::cout << "Device is available of type: " << get_type(dev) << std::endl;
  }
  */
  sycl::queue q_ct1{CUDADeviceSelector()};

  const char *result[2] = {"FAILED", "OK"};
  bool success          = true;
  // Define the kernels granularity: 10K blocks of 32 treads each
  sycl::range<3> nblocks(1, 1, 10000), nthreads(1, 1, 32);

  // Allocate the content of SomeStruct in a buffer
  char *buffer = nullptr;
  buffer        = (char *)sycl::malloc_shared(sizeof(SomeStruct), q_ct1);
  SomeStruct *a = SomeStruct::MakeInstanceAt(buffer);

  // Launch a kernel doing additions
  bool testOK = true;
  std::cout << "   testAdd ... ";
  // Wait memory to reach device
  q_ct1.wait_and_throw();
  /*
  DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
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
  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
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
  /*
  DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
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
