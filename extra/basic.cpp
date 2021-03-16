#include <CL/sycl.hpp>

#include <array>
#include <numeric>
#include <iostream>

int main() {
  /*
  Here we construct a large array of integers and initialize it with the numbers from 0 to array_size-1 
  (this is what std::iota does). Note that we use cl::sycl::cl_int to ensure compatibility.
  */  
  const size_t array_size = 1024*512;
  std::array<cl::sycl::cl_int, array_size> in,out;
  std::iota(begin(in),end(in),0);
  /*
  Next we open up a new scope. This achieves two things:
    - device_queue will be destructed at the end of the scope, which will block until the kernel has completed.
    - in_buffer and out_buffer will also be destructed, which will force data tranfer back to the host and allow 
      us to access the data from in and out.
  */  
  {
    /* 
    Now we create our command queue. The command queue is where all work (kernels) will be enqueued before being 
    dispatched to the device. There are many ways to customise the queue, such as providing a device to enqueue on or 
    setting up asynchronous error handlers, but the default constructor will do for this example; it looks for a compatible 
    GPU and falls back on the host CPU if it fails.
    */
    cl::sycl::queue device_queue;
    std::cout <<  "Running on "
	      << device_queue.get_device().get_info<cl::sycl::info::device::name>()
	      << "\n";
    /*
    Next we create a range, which describes the shape of the data which the kernel will be executing on. In our simple 
    example, it’s a one-dimensional array, so we use cl::sycl::range<1>. If the data was two-dimensional we would use 
    cl::sycl::range<2> and so on. Alongside cl::sycl::range, there is cl::sycl::ndrange, which allows you to specify 
    work group sizes as well as an overall range, but we don’t need that for our example.
    */
    cl::sycl::range<1> n_items{array_size};
    /*
    In order to control data sharing and transfer between the host and devices, SYCL provides a buffer class. We create 
    two SYCL buffers to manage our input and output arrays.
    */  
    cl::sycl::buffer<cl::sycl::cl_int, 1> in_buffer(in.data(), n_items);
    cl::sycl::buffer<cl::sycl::cl_int, 1> out_buffer(out.data(), n_items);
    /*
    After setting up all of our data, we can enqueue our actual work. There are a few ways to do this, but a simple method 
    for setting up a parallel execution is to call the .submit function on our queue. To this function we pass a command 
    group functor2 which will be executed when the runtime schedules that task. A command group handler sets up any last
    resources needed by the kernel and dispatches it.
    */
    device_queue.submit([&](cl::sycl::handler &cgh) {
       constexpr auto sycl_read = cl::sycl::access::mode::read;
       constexpr auto sycl_write = cl::sycl::access::mode::write;
       /*
       In order to control access to our buffers and to tell the runtime how we will be using the data, we need to 
       create accessors. It should be clear that we are creating one accessor for reading from in_buffer, and one accessor 
       for writing to out_buffer.
       */
       auto in_accessor = in_buffer.get_access<sycl_read>(cgh);
       auto out_accessor = out_buffer.get_access<sycl_write>(cgh);
       /*
       Now that we’ve done all the setup, we can actually do some computation on our device. Here we dispatch a kernel on 
       the command group handler cgh over our range n_items. The actual kernel itself is a lambda which takes a work-item 
       identifier and carries out our computation. In this case, we are reading from in_accessor at the index of our work-item 
       identifier, multiplying it by 2, then storing the result in the relevant place in out_accessor. That <class VecScalMul> 
       is an unfortunate byproduct of how SYCL needs to work within the confines of standard C++, so we need to give a unique 
       class name to the kernel for the compiler to be able to do its job.
       */
       cgh.parallel_for<class VecScalMul>(n_items,
	     [=](cl::sycl::id<1> wiID) {
	       out_accessor[wiID] = in_accessor[wiID]*2;
	     });
       });
    /*  
    After this point, our kernel will have completed and we could access out and expect to see the correct results.
    */
  }
}
