#include <CL/sycl.hpp>

extern SYCL_EXTERNAL double stepInField(double kinE, double mass, int charge); // Check field.cu file

#if (defined( __SYCL_DEVICE_ONLY__))
#define log sycl::log
#define exp sycl::exp
#define cos sycl::cos
#define sin sycl::sin
#define pow sycl::pow
#define frexp sycl::frexp
#define ldexp sycl::ldexp
#define modf sycl::modf
#else
#define log std::log
#define exp std::exp
#define cos std::cos
#define sin std::sin
#define pow std::pow
#define frexp std::frexp
#define ldexp std::ldexp
#define modf std::modf
#endif

void kernel(double *step)
{
  // *step =  log(*step);
  // *step =  exp(*step);
  // *step =  cos(*step);
  // *step =  sin(*step);
  // *step =  pow(*step, 1.2);
  // bool flag = std::isinf(*step);
  // double res = std::sqrt(*step);
  // int exponent;
  // frexp(*step, &exponent);
  // ldexp(*step, 3);
  // double iptr;
  // modf(*step, &iptr);
  // *step = fabs(*step);
  *step = stepInField(2.0, 3.0, 1);
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
