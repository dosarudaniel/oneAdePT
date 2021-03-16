#include <CL/sycl.hpp>

class vector_addition;

int main(int, char**) {
    /*
   We begin by setting up host storage for the data that we want to operate on. Our goal is to compute c = a + b, where the variables are vectors. To help us achieve this, SYCL provides the vec<T, size> type, which is a vector of a basic scalar type. It has template parameters for the scalar type and the size. It is meant to be used more like a geometrical vector than std::vector, and so it only supports sizes of up to 16. But dont despair, there are plenty of ways to work on larger sets of data. We use float4, which is just vec<float, 4>.
    */
  cl::sycl::float4 a = { 1.0, 2.0, 3.0, 4.0 };
  cl::sycl::float4 b = { 4.0, 3.0, 2.0, 1.0 };
  cl::sycl::float4 c = { 0.0, 0.0, 0.0, 0.0 };

  /*
   In order to send our tasks to be scheduled and executed on the target device we need to use a SYCL queue. 
   We set this up and pass it our selector so that it knows what device to select when running the tasks.
   */
  cl::sycl::default_selector device_selector;

  cl::sycl::queue queue(device_selector);
  std::cout << "Running on "
	    << queue.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";
  {
    /*
    On most systems, the host and the device do not share physical memory. For example, the CPU might use RAM and the GPU might use its own on-die VRAM. SYCL needs to know which data it will be sharing between the host and the devices.

     For this purpose, SYCL buffers exist. The buffer<T, dims> class is generic over the element type and the number of dimensions, which can be one, two or three. When passed a raw pointer like in this case, the buffer(T* ptr, range size) constructor takes ownership of the memory it has been passed. This means that we absolutely cannot use that memory ourselves while the buffer exists, which is why we begin a C++ scope. At the end of their scope, the buffers will be destroyed and the memory returned to the user. The size argument is a range object, which has to have the same number of dimensions as the buffer and is initialized with the number of elements in each dimension. Here, we have one dimension with one element.

    Buffers are not associated with a particular queue or context, so they are capable of handling data transparently between multiple devices. They also do not require read/write information, as this is specified per operation.
    */
    cl::sycl::buffer<cl::sycl::float4, 1> a_sycl(&a, cl::sycl::range<1>(1));
    cl::sycl::buffer<cl::sycl::float4, 1> b_sycl(&b, cl::sycl::range<1>(1));
    cl::sycl::buffer<cl::sycl::float4, 1> c_sycl(&c, cl::sycl::range<1>(1));

    /*
The whole thing is technically a single function call to queue::submit. submit accepts a function object parameter, which encapsulates a command group. For this purpose, the function object accepts a command group handler constructed by the SYCL runtime and handed to us as the argument. All operations using a given command group handler are part of the same command group.

A command group is a way to encapsulate a device-side operation and all its data dependencies in a single object by grouping all the related commands (function calls). Effectively, what this achieves is preventing data race conditions, resource leaking and other problems by letting the SYCL runtime know the prerequisites for executing device-side code correctly.

     */
    queue.submit([&] (cl::sycl::handler& cgh) {
		   /*
In our command group, we first setup accessors. In general, these objects define the inputs and outputs of a device-side operation. The accessors also provide access to various forms of memory. In this case, they allow us to access the memory owned by the buffers created earlier. We passed ownership of our data to the buffer, so we can no longer use the float4 objects, and accessors are the only way to access data in buffer objects.

The buffer::get_access(handler&) method takes an access mode argument. We use access::mode::read for the arguments and access::mode::discard_write for the result. discard_write can be used whenever we write to the whole buffer and do not care about its previous contents. Since it will be overwritten entirely, we can discard whatever was there before.

The second parameter is the type of memory we want to access the data from. We will see the available types of memory in the section on memory accesses. For now we use the default value.
		    */
		   auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
		   auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
		   auto c_acc = c_sycl.get_access<cl::sycl::access::mode::discard_write>(cgh);
		   /*
In SYCL there are various ways to define a kernel function that will execute on a device depending on the kind of parallelism you want and the different features you require. The simplest of these is the cl::sycl::handler::single_task function, which takes a single parameter, being a C++ function object and executes that function object exactly once on the device. The C++ function object does not take any parameters, however it is important to note that if the function object is a lambda it must capture by value and if it is a struct or class it must define all members as value members.
		    */
		   cgh.single_task<class vector_addition>([=] () {
							    c_acc[0] = a_acc[0] + b_acc[0];
							  });
		 });
    /*
One of the features of SYCL is that it makes use of C++ RAII (resource aquisition is initialisation), meaning that there is no explicit cleanup and everything is done via the SYCL object destructors.
     */
  }
  std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
            << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
            << "------------------\n"
            << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }"
            << std::endl;

  return 0;
}
