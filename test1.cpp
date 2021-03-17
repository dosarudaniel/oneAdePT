// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <CopCore/1/Ranluxpp.h>

#include <iostream>

#include <assert.h>
#include <stdlib.h>

void __assert_fail(const char * assertion, const char * file, unsigned int line, const char * function)
{
 
}

void kernel(RanluxppDouble *r, double *d, uint64_t *i, double *d2)
{
  *d = r->Rndm();
  *i = r->IntRndm();
  r->Skip(42);
  *d2 = r->Rndm();
}

int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  RanluxppDouble r;
  std::cout << "double: " << r.Rndm() << std::endl;
  std::cout << "int: " << r.IntRndm() << std::endl;

  RanluxppDouble *r_dev;

  r_dev = sycl::malloc_device<RanluxppDouble>(1, q_ct1);

  double *d_dev_ptr;
  uint64_t *i_dev_ptr;
  double *d2_dev_ptr;

  d_dev_ptr  = sycl::malloc_device<double>(1, q_ct1);
  i_dev_ptr  = sycl::malloc_device<uint64_t>(1, q_ct1);
  d2_dev_ptr = sycl::malloc_device<double>(1, q_ct1);

  // Transfer the state of the generator to the device.
  q_ct1.memcpy(r_dev, &r, sizeof(RanluxppDouble)).wait();

  //dev_ct1.queues_wait_and_throw();
  q_ct1.wait_and_throw();

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       kernel(r_dev, d_dev_ptr, i_dev_ptr, d2_dev_ptr);
                     });
  });

  q_ct1.wait_and_throw();
  //dev_ct1.queues_wait_and_throw();

  // Generate from the same state on the host.
  double d   = r.Rndm();
  uint64_t i = r.IntRndm();
  r.Skip(42);
  double d2 = r.Rndm();

  // Fetch the numbers from the device for comparison.
  double d_dev;
  uint64_t i_dev;
  double d2_dev;

  q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();
  q_ct1.memcpy(&i_dev, i_dev_ptr, sizeof(uint64_t)).wait();
  q_ct1.memcpy(&d2_dev, d2_dev_ptr, sizeof(double)).wait();

  //  dev_ct1.queues_wait_and_throw();
  q_ct1.wait_and_throw();

  int ret = 0;

  std::cout << std::endl;
  std::cout << "double:" << std::endl;
  std::cout << "   host:   " << d << std::endl;
  std::cout << "   device: " << d_dev << std::endl;

  ret += (d != d_dev);

  std::cout << "int:" << std::endl;
  std::cout << "   host:   " << i << std::endl;
  std::cout << "   device: " << i_dev << std::endl;
  ret += (i != i_dev);

  std::cout << "double (after calling Skip(42)):" << std::endl;
  std::cout << "   host:   " << d2 << std::endl;
  std::cout << "   device: " << d2_dev << std::endl;
  ret += (d2 != d2_dev);

  sycl::free(r_dev, q_ct1);
  sycl::free(d_dev_ptr, q_ct1);
  sycl::free(i_dev_ptr, q_ct1);
  sycl::free(d2_dev_ptr, q_ct1);

  return ret;
}
