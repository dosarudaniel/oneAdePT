// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/* Use this command to avoid cmake: 

/home/dadosaru/sycl_workspace/llvm/build//bin/clang++ ../../tests/test10.cpp -o test10 -lsycl -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -Wno-unknown-cuda-version -D_ONEADEPT_ -DROOT_NO_INT128 -DG4HepEm_CUDA_BUILD --verbose
*/


#include <CL/sycl.hpp>

#include <iostream>
#include <stdlib.h>
#include <math.h>

void kernel(double *d)
{
  *d = sqrt(*d); // passes this point

  *d = log(*d); // fails here
}


int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on " << q_ct1.get_device().get_info<cl::sycl::info::device::name>() << "\n";
 
  double value = 10;
  double *d_dev_ptr = &value;

  d_dev_ptr  = sycl::malloc_device<double>(1, q_ct1);

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       kernel(d_dev_ptr);
                     });
  });

  q_ct1.wait_and_throw();
 
  double d_dev;

  q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();

  q_ct1.wait_and_throw();

  double d = std::log(value);
  
  std::cout << std::endl;
  std::cout << "double:" << std::endl;
  std::cout << "   host:   " << d << std::endl;
  std::cout << "   device: " << d_dev << std::endl;

  sycl::free(d_dev_ptr, q_ct1);

}
