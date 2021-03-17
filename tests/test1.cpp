#include <iostream>
#include <CL/sycl.hpp>
#include <oneapi/mkl/rng/device.hpp>



void select_process(oneapi::mkl::rng::device::philox4x32x10<1> *engine, sycl::item<1>, float *res)
{
    oneapi::mkl::rng::device::uniform<> distr;
   // generate random number
    *res = oneapi::mkl::rng::device::generate(distr, *engine);
}

int main() {
    sycl::queue queue;
    const int n = 1000;
    const int seed = 0;
    // Prepare an array for random numbers
    std::vector<float> r(n);

    using rng_t = oneapi::mkl::rng::device::philox4x32x10<1>;

    rng_t engine(seed);
    rng_t *state = malloc_shared<rng_t>(1,queue);
    queue.memcpy(state, &engine, sizeof(rng_t)).wait();


    sycl::buffer<float, 1> r_buf(r.data(), r.size());
    // Submit a kernel to generate on device
    queue.submit([&](sycl::handler& cgh) {
        auto r_acc = r_buf.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::item<1> item) {
            float res;
            //select_process(&engine,item,&res);
            select_process(state,item,&res);
            r_acc[item.get_id(0)] = res;
        });
    });

    auto r_acc = r_buf.template get_access<sycl::access::mode::read>();
    std::cout << "Samples of uniform distribution" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << r_acc[i] << std::endl;
    }

    return 0;
}

