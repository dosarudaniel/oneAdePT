
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <oneapi/mkl/rng/device.hpp>

// example parameters defines
#define SEED    0
#define N       1024
#define N_PRINT 10


int main (int argc, char ** argv) {

    sycl::default_selector device_selector;
    sycl::queue queue(device_selector);
    std::cout <<  "Running on "
	    << queue.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

    const int seed = 1;

    const sycl::range nthreads(32);
    const sycl::range maxBlocks(10);

    // prepare array for random numbers

    std::vector<float> r_host(N);

    float* r_dev = (float*)sycl::malloc_shared(sizeof(float) * N, queue);

    sycl::range<1> range(N);
    
     using rnd_t = oneapi::mkl::rng::device::philox4x32x10<1>;
     rnd_t* states = (rnd_t*) sycl::malloc_device(sizeof(rnd_t) * (nthreads.get(0) *  maxBlocks.get(0)), queue);

    // submit kernels to init engines and generate on device
    try {
        // kernel with initialization of rng engine
      auto event = queue.submit([&](sycl::handler& cgh) {
		      cgh.parallel_for(sycl::range<1>(maxBlocks * nthreads), [=](sycl::item<1> item) {
	                         size_t id =  item.get_id(0);
                             rnd_t engine(seed, id );
                             states[id] = engine;
				});
	   });
        event.wait();
        // generate random numbers
        event = queue.submit([&](sycl::handler &cgh) {
            /*
                cgh.parallel_for(sycl::range<1>(maxBlocks * nthreads), [=](sycl::item<1> item) {
                   	size_t id =  item.get_id(0);
                    oneapi::mkl::rng::device::uniform<float> distr;
                    auto res = oneapi::mkl::rng::device::generate(distr, states[id]);
                    r_dev[id] = res;
	            });
            */
		        cgh.parallel_for(sycl::nd_range<1>(maxBlocks * nthreads, nthreads), [=](sycl::nd_item<1> item) {
	            size_t id =  item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
                oneapi::mkl::rng::device::uniform<float> distr;
                auto res = oneapi::mkl::rng::device::generate(distr, states[id]);
                r_dev[id] = res;
            });
        });
											   
   /*	
        event = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                oneapi::mkl::rng::device::uniform<float> distr;
                auto res = oneapi::mkl::rng::device::generate(distr, states[id]);
                r_dev[id] = res;
            });
        });
   */
        queue.wait_and_throw();
    }
    catch(sycl::exception const& e) {
        std::cout << "\t\tSYCL exception\n"
                    << e.what() << std::endl;
        return 1;
    }

    std::cout << "\t\tOutput of generator:" << std::endl;

    std::cout << "first "<< N_PRINT << " numbers of " << N << ": " << std::endl;
    for(int i = 0 ; i < N_PRINT; i++) {
        std::cout << r_dev[i] << " ";
    }
    std::cout << std::endl;

    // compare results with host-side generation
    oneapi::mkl::rng::device::philox4x32x10<> engine(seed);
    oneapi::mkl::rng::device::uniform<float> distr;

    int err = 0;
    for(int i = 0; i < N; i++) {
        r_host[i] = oneapi::mkl::rng::device::generate(distr, engine);
        if(r_host[i] != r_dev[i]) {
            std::cout << "error in " << i << " element " << r_host[i] << " " << r_dev[i] << std::endl;
            err++;
        }
    }
    sycl::free(states, queue);
    return err;
}

