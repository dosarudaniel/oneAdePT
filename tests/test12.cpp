#include <CL/sycl.hpp>

extern SYCL_EXTERNAL double stepInField(double kinE, double mass, int charge); // Check field.cu file

void kernel(double *step)
{
  #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
     *step = stepInField(2.0, 3.0, 1);
  #endif
}

void kernel2()
{

}

class scalar_add;


int main(void)
{
  
  int a = 18, b = 24, r = 0;

  auto defaultQueue = sycl::queue{};

  {
    auto bufA = sycl::buffer{&a, sycl::range{1}};
    auto bufB = sycl::buffer{&b, sycl::range{1}};
    auto bufR = sycl::buffer{&r, sycl::range{1}};

    defaultQueue
        .submit([&](sycl::handler &cgh) {
          auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
          auto accB = sycl::accessor{bufB, cgh, sycl::read_only};
          auto accR = sycl::accessor{bufR, cgh, sycl::write_only};

          cgh.single_task<scalar_add>([=] { accR[0] = accA[0] + accB[0]; });
        })
        .wait();
  }
  
  // std::cout <<  "Running on "
  //       	<< q_ct1.get_device().get_info<cl::sycl::info::device::name>()
  //       	<< "\n";
 
  // double *d_dev_ptr;

  // d_dev_ptr  = sycl::malloc_shared<double>(1, q_ct1);

  // q_ct1.submit([&](sycl::handler &cgh) {
  //   cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
  //     sycl::range<3>(1, 1, 1)), 
  //     [=](sycl::nd_item<3> item_ct1) {
  //       kernel(d_dev_ptr);
  //   });
  // }).wait();

  // // q_ct1.submit([&](sycl::handler &cgh) {
  // //   cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
  // //                                       sycl::range<3>(1, 1, 1)), 
  // //                   [=](sycl::nd_item<3> item_ct1) {
  // //     kernel2();
  // //   });
  // // }).wait();

  // double d_dev;
  // q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();
  // std::cout << "   device: " << d_dev << std::endl;
  // sycl::free(d_dev_ptr, q_ct1);
}

