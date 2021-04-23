#include <CL/sycl.hpp>

extern SYCL_EXTERNAL double stepInField(double kinE, double mass, int charge); // Check field.cu file

void kernel(double *step)
{
  *step =  stepInField(6.0f, 7.0f,1);
}

int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
        	<< q_ct1.get_device().get_info<cl::sycl::info::device::name>()
        	<< "\n";

 
  double *d_dev_ptr;
  d_dev_ptr  = sycl::malloc_device<double>(1, q_ct1);

  q_ct1.submit([&](sycl::handler &cgh) {
	cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
  sycl::range<3>(1, 1, 1)),
                 	[=](sycl::nd_item<3> item_ct1) {
                   	kernel(d_dev_ptr);
                 	});
  }).wait();
 
  double d_dev;
  q_ct1.memcpy(&d_dev, d_dev_ptr, sizeof(double)).wait();

  std::cout << "   device: " << d_dev << std::endl;

  sycl::free(d_dev_ptr, q_ct1);
}
