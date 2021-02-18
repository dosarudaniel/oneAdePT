/*******************************************************************************
* Copyright 2018-2020 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, float alpha,
                           const float *x, std::int64_t incx, std::int64_t stridex,
                           float *y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, double alpha,
                           const double *x, std::int64_t incx, std::int64_t stridex,
                           double *y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
    
DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                           const std::complex<float> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<float> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
    
DLL_EXPORT cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                           const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<double> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

// Level 3

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda,
                           const float **b, std::int64_t *ldb, 
                           float *beta, float **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *groupsize,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda,
                           const double **b, std::int64_t *ldb, 
                           double *beta, double **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *groupsize,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, std::int64_t *lda,
                           const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *groupsize,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, std::int64_t *lda,
                           const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *groupsize,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, half *alpha,
                           const half **a, std::int64_t *lda,
                           const half **b, std::int64_t *ldb, 
                           half *beta, half **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *groupsize,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a,
                           const float *b, std::int64_t ldb, std::int64_t stride_b,
                           float beta, float *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b,
                           double beta, double *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
                           const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
                           std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
                           const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
                           std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                           const half *a, std::int64_t lda, std::int64_t stride_a,
                           const half *b, std::int64_t ldb, std::int64_t stride_b,
                           half beta, half *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, float alpha, const float *a, std::int64_t lda,
              const float *b, std::int64_t ldb, float beta, float *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
    
DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, double alpha, const double *a, std::int64_t lda,
              const double *b, std::int64_t ldb, double beta, double *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
    
DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, half alpha, const half *a, std::int64_t lda,
              const half *b, std::int64_t ldb, half beta, half *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, float alpha, const half *a, std::int64_t lda,
              const half *b, std::int64_t ldb, float beta, float *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, float alpha, const bfloat16 *a, std::int64_t lda,
              const bfloat16 *b, std::int64_t ldb, float beta, float *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              float alpha, const float *a, std::int64_t lda,
              const float *b, std::int64_t ldb, float beta, float *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              double alpha, const double *a, std::int64_t lda,
              const double *b, std::int64_t ldb, double beta, double *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *b, std::int64_t ldb,
              std::complex<float> beta, std::complex<float> *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *b, std::int64_t ldb,
              std::complex<double> beta, std::complex<double> *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *b, std::int64_t ldb,
              std::complex<float> beta, std::complex<float> *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *b, std::int64_t ldb,
              std::complex<double> beta, std::complex<double> *c,
              std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              float alpha, const float *a, std::int64_t lda,
              float beta, float *c, std::int64_t ldc,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              double alpha, const double *a, std::int64_t lda,
              double beta, double *c, std::int64_t ldc,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
              std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
              std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              float alpha, const std::complex<float> *a, std::int64_t lda,
              float beta, std::complex<float> *c, std::int64_t ldc,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              double alpha, const std::complex<double> *a, std::int64_t lda,
              double beta, std::complex<double> *c, std::int64_t ldc,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               float alpha, const float *a, std::int64_t lda,
               const float *b, std::int64_t ldb,
               float beta, float *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               double alpha, const double *a, std::int64_t lda,
               const double *b, std::int64_t ldb,
               double beta, double *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
               const std::complex<float> *b, std::int64_t ldb,
               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
               const std::complex<double> *b, std::int64_t ldb,
               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
               const std::complex<float> *b, std::int64_t ldb,
               float beta, std::complex<float> *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
               const std::complex<double> *b, std::int64_t ldb,
               double beta, std::complex<double> *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              float alpha, const float *a, std::int64_t lda,
              float *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              double alpha, const double *a, std::int64_t lda,
              double *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
              std::complex<float> *b, std::int64_t ldb,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
              std::complex<double> *b, std::int64_t ldb,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, 
                           transpose trans, diag unit_diag,
                           int64_t m, int64_t n,
                           float alpha, const float *a, 
                           int64_t lda, int64_t stride_a,
                           float *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, 
                           transpose trans, diag unit_diag,
                           int64_t m, int64_t n,
                           double alpha, const double *a, 
                           int64_t lda, int64_t stride_a,
                           double *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, 
                           transpose trans, diag unit_diag,
                           int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, 
                           int64_t lda, int64_t stride_a,
                           std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, 
                           transpose trans, diag unit_diag,
                           int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, 
                           int64_t lda, int64_t stride_a,
                           std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower, 
                           transpose *trans, diag *unit_diag,
                           int64_t *m, int64_t *n,
                           float *alpha, const float **a, int64_t *lda,
                           float **b, int64_t *ldb, 
                           int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower, 
                           transpose *trans, diag *unit_diag,
                           int64_t *m, int64_t *n,
                           double *alpha, const double **a, int64_t *lda,
                           double **b, int64_t *ldb, 
                           int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower, 
                           transpose *trans, diag *unit_diag,
                           int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           std::complex<float> **b, int64_t *ldb, 
                           int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower, 
                           transpose *trans, diag *unit_diag,
                           int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a, int64_t *lda,
                           std::complex<double> **b, int64_t *ldb, 
                           int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              float alpha, const float *a, std::int64_t lda,
              float *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              double alpha, const double *a, std::int64_t lda,
              double *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
     
DLL_EXPORT cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
              std::complex<float> *b, std::int64_t ldb,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
              std::complex<double> *b, std::int64_t ldb,
              const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

// Level 2

DLL_EXPORT cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
              const float *a, std::int64_t lda,
              const float *x, std::int64_t incx, float beta,
              float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});


DLL_EXPORT cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
              const double *a, std::int64_t lda,
              const double *x, std::int64_t incx, double beta,
              double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
              std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
              std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

    
DLL_EXPORT cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
              const float *a, std::int64_t lda,
              const float *x, std::int64_t incx, float beta,
              float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
              const double *a, std::int64_t lda,
              const double *x, std::int64_t incx, double beta,
              double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
              std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
              const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
              std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
              std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
              const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
              std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
             const float *x, std::int64_t incx, const float *y, std::int64_t incy,
             float *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
             const double *x, std::int64_t incx, const double *y, std::int64_t incy,
             double *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *x, std::int64_t incx,
              const std::complex<float> *y, std::int64_t incy,
              std::complex<float> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *x, std::int64_t incx,
              const std::complex<double> *y, std::int64_t incy,
              std::complex<double> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *x, std::int64_t incx,
              const std::complex<float> *y, std::int64_t incy,
              std::complex<float> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *x, std::int64_t incx,
              const std::complex<double> *y, std::int64_t incy,
              std::complex<double> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, std::complex<float> alpha,
              const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
              std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, std::complex<double> alpha,
              const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
              std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *a, std::int64_t lda,
              const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
              std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *a, std::int64_t lda,
              const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
              std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const std::complex<float> *x, std::int64_t incx,
              std::complex<float> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             const std::complex<double> *x, std::int64_t incx,
             std::complex<double> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *x, std::int64_t incx,
              const std::complex<float> *y, std::int64_t incy,
              std::complex<float> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *x, std::int64_t incx,
              const std::complex<double> *y, std::int64_t incy,
              std::complex<double> *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *a,
              const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
              std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *a,
              const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
              std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const std::complex<float> *x, std::int64_t incx,
              std::complex<float> *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             const std::complex<double> *x, std::int64_t incx,
             std::complex<double> *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              const std::complex<float> *x, std::int64_t incx,
              const std::complex<float> *y, std::int64_t incy,
              std::complex<float> *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              const std::complex<double> *x, std::int64_t incx,
              const std::complex<double> *y, std::int64_t incy,
              std::complex<double> *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, float alpha,
              const float *a, std::int64_t lda,
              const float *x, std::int64_t incx, float beta,
              float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, double alpha,
              const double *a, std::int64_t lda,
              const double *x, std::int64_t incx, double beta,
              double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const float *a, std::int64_t lda,
              const float *x, std::int64_t incx, float beta,
              float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              const double *a, std::int64_t lda,
              const double *x, std::int64_t incx, double beta,
              double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const float *x, std::int64_t incx,
              float *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             const double *x, std::int64_t incx,
             double *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const float *x, std::int64_t incx,
              const float *y, std::int64_t incy,
              float *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              const double *x, std::int64_t incx,
              const double *y, std::int64_t incy,
              double *a, std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const float *a,
              const float *x, std::int64_t incx, float beta,
              float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              const double *a,
              const double *x, std::int64_t incx, double beta,
              double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const float *x, std::int64_t incx,
              float *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             const double *x, std::int64_t incx,
             double *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              const float *x, std::int64_t incx,
              const float *y, std::int64_t incy,
              float *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              const double *x, std::int64_t incx,
              const double *y, std::int64_t incy,
              double *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const float *a, std::int64_t lda,
              float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const double *a, std::int64_t lda,
              double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const std::complex<float> *a, std::int64_t lda,
              std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const std::complex<double> *a, std::int64_t lda,
              std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const float *a, std::int64_t lda,
              float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const double *a, std::int64_t lda,
              double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const std::complex<float> *a, std::int64_t lda,
              std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              const std::complex<double> *a, std::int64_t lda,
              std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const float *a,
              float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const double *a,
              double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<float> *a,
              std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<double> *a,
              std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const float *a,
              float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const double *a,
              double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<float> *a,
              std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<double> *a,
              std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const float *a, std::int64_t lda,
              float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const double *a, std::int64_t lda,
              double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<float> *a, std::int64_t lda,
              std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<double> *a, std::int64_t lda,
              std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const float *a, std::int64_t lda,
              float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const double *a, std::int64_t lda,
              double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<float> *a, std::int64_t lda,
              std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              const std::complex<double> *a, std::int64_t lda,
              std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

// Level 1

DLL_EXPORT cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
              std::complex<float> *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
              std::complex<double> *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
              std::complex<float> *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
              std::complex<double> *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
               std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx, float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx, double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x, std::int64_t incx, float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x, std::int64_t incx, double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y, std::int64_t incy, double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx, float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx, double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c, double s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y, std::int64_t incy, float c, float s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y, std::int64_t incy, double c, double s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b, float *c, std::complex<float> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b, double *c, std::complex<double> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y, std::int64_t incy, float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y, std::int64_t incy, double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1, float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1, double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, double *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x, std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

// BLAS-like extension

DLL_EXPORT cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, float alpha, const float *a, std::int64_t lda,
               const float *b, std::int64_t ldb, float beta, float *c,
               std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, double alpha, const double *a, std::int64_t lda,
               const double *b, std::int64_t ldb, double beta, double *c,
               std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
               const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
               std::complex<float> *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
               const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
               std::complex<double> *c, std::int64_t ldc,
               const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                         offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                         const std::int8_t *a, int64_t lda, std::int8_t ao,
                         const std::uint8_t *b, int64_t ldb, std::uint8_t bo,
                         float beta, std::int32_t *c, int64_t ldc, const std::int32_t *co,
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                         offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                         const std::int8_t *a, int64_t lda, std::int8_t ao,
                         const std::int8_t *b, int64_t ldb, std::int8_t bo,
                         float beta, std::int32_t *c, int64_t ldc, const std::int32_t *co,
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                         offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                         const std::uint8_t *a, int64_t lda, std::uint8_t ao,
                         const std::int8_t *b, int64_t ldb, std::int8_t bo,
                         float beta, std::int32_t *c, int64_t ldc, const std::int32_t *co,
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

DLL_EXPORT cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                         offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                         const std::uint8_t *a, int64_t lda, std::uint8_t ao,
                         const std::uint8_t *b, int64_t ldb, std::uint8_t bo,
                         float beta, std::int32_t *c, int64_t ldc, const std::int32_t *co,
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

