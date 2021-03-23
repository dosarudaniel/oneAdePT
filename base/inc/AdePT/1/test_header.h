
#ifdef __SYCL_DEVICE_ONLY__
  #define __constant__ 
  #define VECGEOM_DEVICE_COMPILATION
  #define VECCORE_CUDA_DEVICE_COMPILATION
  #pragma message("__SYCL_DEVICE_ONLY__")
#else
  #pragma message("NOT __SYCL_DEVICE_ONLY__")
#endif


