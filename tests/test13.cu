// Adapted from https://github.com/codeplaysoftware/dpcpp-workshop/blob/master/exercises/Exercise3_Calling_CUDA_kernels_in_SYCL/solution.cu


#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
 
__global__ void stepInField(double *res, double *kinE, double *mass, int *charge, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) {
        res[id] = kinE[id]*mass[id]*charge[id];
    }
}
 
int main(void)
{
    sycl::default_selector device_selector;

    sycl::queue q_ct1(device_selector);
    std::cout <<  "Running on "
            << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

    int N = 1;
 
    sycl::buffer<double> bRes{sycl::range<1>(N)};
    sycl::buffer<double> bkinE{sycl::range<1>(N)};
    sycl::buffer<double> bMass{sycl::range<1>(N)};
    sycl::buffer<int> bCharge{sycl::range<1>(N)};

    {
        auto hRes = bRes.get_access<sycl::access::mode::write>();
        auto hkinE = bkinE.get_access<sycl::access::mode::write>();
        auto hMass = bMass.get_access<sycl::access::mode::write>();
        auto hCharge = bCharge.get_access<sycl::access::mode::write>();

        // Initialize vectors on host
        for (int i = 0; i < N; i++) {
            hRes[i] = 0;
            hkinE[i] = 3.0;
            hMass[i] = 2.0;
            hCharge[i] = 1;
        }
    }

    q_ct1.submit([&](sycl::handler &cgh) {

        auto accRes = bRes.get_access<sycl::access::mode::write>(cgh);
        auto acckinE = bkinE.get_access<sycl::access::mode::read>(cgh);
        auto accMass = bMass.get_access<sycl::access::mode::read>(cgh);
        auto accCharge = bCharge.get_access<sycl::access::mode::read>(cgh);

        cgh.interop_task([=](sycl::interop_handler ih) {
            auto res = reinterpret_cast<double *>(ih.get_mem<sycl::backend::cuda>(accRes));
            auto kinE = reinterpret_cast<double *>(ih.get_mem<sycl::backend::cuda>(acckinE));
            auto Mass = reinterpret_cast<double *>(ih.get_mem<sycl::backend::cuda>(accMass));
            auto Charge = reinterpret_cast<int *>(ih.get_mem<sycl::backend::cuda>(accCharge));

            stepInField<<<1,1>>>(res, kinE, Mass, Charge, N);
        });

    }).wait();

    auto hRes = bRes.get_access<sycl::access::mode::read>();

    for (int i = 0; i < N; i++) {
        std::cout << "result = " << hRes[i] << " ( Expected: 6 )" << std::endl;
    }
 }
