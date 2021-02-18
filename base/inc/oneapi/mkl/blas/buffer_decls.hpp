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

// Level 3

DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
              cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              std::int64_t ldc);

DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
              cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
              std::int64_t ldc);

DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);


DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, std::int64_t lda,
              cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c,
              std::int64_t ldc);

DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, float alpha, cl::sycl::buffer<half, 1> &a, std::int64_t lda,
              cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              std::int64_t ldc);

DLL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
              std::int64_t k, float alpha, cl::sycl::buffer<bfloat16, 1> &a, std::int64_t lda,
              cl::sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              std::int64_t ldc);

DLL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
              cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              std::int64_t ldc);


DLL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
              cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
              std::int64_t ldc);


DLL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
              std::int64_t ldc);


DLL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              std::int64_t ldc);


DLL_EXPORT void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
              std::int64_t ldc);


DLL_EXPORT void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
              std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              std::int64_t ldc);


DLL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
              float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);


DLL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
              double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);


DLL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);


DLL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);


DLL_EXPORT void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              float beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);


DLL_EXPORT void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
              double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              double beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);


DLL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
               cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
               float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);


DLL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
               cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
               double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);


DLL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
               std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);


DLL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
               std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);


DLL_EXPORT void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
               float beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);


DLL_EXPORT void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
               std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
               double beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);


DLL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
              cl::sycl::buffer<float, 1> &b, std::int64_t ldb);


DLL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
              cl::sycl::buffer<double, 1> &b, std::int64_t ldb);


DLL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);


DLL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);


DLL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
              cl::sycl::buffer<float, 1> &b, std::int64_t ldb);


DLL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
              cl::sycl::buffer<double, 1> &b, std::int64_t ldb);


DLL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);


DLL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t m, std::int64_t n,
              std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

// Level 2


DLL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx, float beta,
              cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx, double beta,
              cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, std::complex<float> beta,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, std::complex<double> beta,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx, float beta,
              cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx, double beta,
              cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
              std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, std::complex<float> beta,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
              std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, std::complex<double> beta,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
             cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy,
             cl::sycl::buffer<float,1> &a, std::int64_t lda);


DLL_EXPORT void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
             cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy,
             cl::sycl::buffer<double,1> &a, std::int64_t lda);


DLL_EXPORT void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda);


DLL_EXPORT void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda);


DLL_EXPORT void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda);


DLL_EXPORT void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda);


DLL_EXPORT void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, std::complex<float> beta,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, std::complex<double> beta,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, std::complex<float> beta,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, std::complex<double> beta,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda);


DLL_EXPORT void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx,
             cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda);


DLL_EXPORT void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda);


DLL_EXPORT void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda);


DLL_EXPORT void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &a,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, std::complex<float> beta,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &a,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, std::complex<double> beta,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<float>,1> &a);


DLL_EXPORT void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx,
             cl::sycl::buffer<std::complex<double>,1> &a);


DLL_EXPORT void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<float>,1> &a);


DLL_EXPORT void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx,
              cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<double>,1> &a);


DLL_EXPORT void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx, float beta,
              cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, double alpha,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx, double beta,
              cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx, float beta,
              cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx, double beta,
              cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &x, std::int64_t incx,
              cl::sycl::buffer<float,1> &a, std::int64_t lda);


DLL_EXPORT void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             cl::sycl::buffer<double,1> &x, std::int64_t incx,
             cl::sycl::buffer<double,1> &a, std::int64_t lda);


DLL_EXPORT void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &x, std::int64_t incx,
              cl::sycl::buffer<float,1> &y, std::int64_t incy,
              cl::sycl::buffer<float,1> &a, std::int64_t lda);


DLL_EXPORT void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              cl::sycl::buffer<double,1> &x, std::int64_t incx,
              cl::sycl::buffer<double,1> &y, std::int64_t incy,
              cl::sycl::buffer<double,1> &a, std::int64_t lda);


DLL_EXPORT void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &a,
              cl::sycl::buffer<float,1> &x, std::int64_t incx, float beta,
              cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              cl::sycl::buffer<double,1> &a,
              cl::sycl::buffer<double,1> &x, std::int64_t incx, double beta,
              cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &x, std::int64_t incx,
              cl::sycl::buffer<float,1> &a);


DLL_EXPORT void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
             cl::sycl::buffer<double,1> &x, std::int64_t incx,
             cl::sycl::buffer<double,1> &a);


DLL_EXPORT void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
              cl::sycl::buffer<float,1> &x, std::int64_t incx,
              cl::sycl::buffer<float,1> &y, std::int64_t incy,
              cl::sycl::buffer<float,1> &a);


DLL_EXPORT void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
              cl::sycl::buffer<double,1> &x, std::int64_t incx,
              cl::sycl::buffer<double,1> &y, std::int64_t incy,
              cl::sycl::buffer<double,1> &a);


DLL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n, std::int64_t k,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<float,1> &a,
              cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<double,1> &a,
              cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<float>,1> &a,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<double>,1> &a,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<float,1> &a,
              cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<double,1> &a,
              cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<float>,1> &a,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<double>,1> &a,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<float,1> &a, std::int64_t lda,
              cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<double,1> &a, std::int64_t lda,
              cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<float>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
              std::int64_t n,
              cl::sycl::buffer<std::complex<double>,1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);

// Level 1


DLL_EXPORT void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x,
              std::int64_t incx, cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<float>,1> &result);


DLL_EXPORT void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x,
              std::int64_t incx, cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<double>,1> &result);


DLL_EXPORT void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x,
              std::int64_t incx, cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<float>,1> &result);


DLL_EXPORT void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x,
              std::int64_t incx, cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy,
              cl::sycl::buffer<std::complex<double>,1> &result);


DLL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
               cl::sycl::buffer<std::int64_t, 1> &result);


DLL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &result);


DLL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &result);


DLL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &result);


DLL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &result);


DLL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);


DLL_EXPORT void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy, cl::sycl::buffer<float,1> &result);


DLL_EXPORT void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy, cl::sycl::buffer<double,1> &result);


DLL_EXPORT void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy, cl::sycl::buffer<float,1> &result);


DLL_EXPORT void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy, cl::sycl::buffer<double,1> &result);


DLL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &result);


DLL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &result);


DLL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &result);


DLL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &result);


DLL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy, float c, float s);


DLL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy, double c, double s);


DLL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy, float c, float s);


DLL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy, double c, double s);


DLL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float,1> &a, cl::sycl::buffer<float,1> &b, cl::sycl::buffer<float,1> &c, cl::sycl::buffer<float,1> &s);


DLL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double,1> &a, cl::sycl::buffer<double,1> &b, cl::sycl::buffer<double,1> &c, cl::sycl::buffer<double,1> &s);


DLL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>,1> &a, cl::sycl::buffer<std::complex<float>,1> &b, cl::sycl::buffer<float,1> &c, cl::sycl::buffer<std::complex<float>,1> &s);


DLL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>,1> &a, cl::sycl::buffer<std::complex<double>,1> &b, cl::sycl::buffer<double,1> &c, cl::sycl::buffer<std::complex<double>,1> &s);


DLL_EXPORT void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy, cl::sycl::buffer<float,1> &param);


DLL_EXPORT void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy, cl::sycl::buffer<double,1> &param);


DLL_EXPORT void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float,1> &d1, cl::sycl::buffer<float,1> &d2, cl::sycl::buffer<float,1> &x1, float y1, cl::sycl::buffer<float,1> &param);


DLL_EXPORT void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double,1> &d1, cl::sycl::buffer<double,1> &d2, cl::sycl::buffer<double,1> &x1, double y1, cl::sycl::buffer<double,1> &param);


DLL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float,1> &x, std::int64_t incx);


DLL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double,1> &x, std::int64_t incx);


DLL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx);


DLL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx);


DLL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float,1> &x, std::int64_t incx, cl::sycl::buffer<float,1> &y, std::int64_t incy);


DLL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double,1> &x, std::int64_t incx, cl::sycl::buffer<double,1> &y, std::int64_t incy);


DLL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy);


DLL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy);

// Batch API

DLL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
                    std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                    std::int64_t stride_a,
                    cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta, cl::sycl::buffer<float, 1> &c,
                    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);


DLL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
                    std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                    std::int64_t stride_a,
                    cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b, double beta, cl::sycl::buffer<double, 1> &c,
                    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);


DLL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
                    std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                    std::int64_t stride_a,
                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);


DLL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
                    std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                    std::int64_t stride_a,
                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

DLL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
                    std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                    std::int64_t stride_a,
                    cl::sycl::buffer<half, 1> &b, std::int64_t ldb, std::int64_t stride_b, half beta, cl::sycl::buffer<half, 1> &c,
                    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size); 

DLL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
                    std::int64_t m, std::int64_t n,
                    float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                    cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

DLL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
                    std::int64_t m, std::int64_t n,
                    double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                    cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

DLL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
                    std::int64_t m, std::int64_t n,
                    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::int64_t stride_a,
                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);


DLL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
                    std::int64_t m, std::int64_t n,
                    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::int64_t stride_a,
                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

DLL_EXPORT void axpy_batch(cl::sycl::queue &queue, std::int64_t n, float alpha,
                cl::sycl::buffer<float,1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<float,1> &y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

DLL_EXPORT void axpy_batch(cl::sycl::queue &queue, std::int64_t n, double alpha,
                cl::sycl::buffer<double,1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<double,1> &y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

DLL_EXPORT void axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>,1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<std::complex<float>,1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

DLL_EXPORT void axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>,1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<std::complex<double>,1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

// BLAS like

DLL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
               cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
               std::int64_t ldc);

DLL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
               cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
               std::int64_t ldc);

DLL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

DLL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
               std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

DLL_EXPORT void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                  std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                  cl::sycl::buffer<std::int8_t, 1> &a, std::int64_t lda, std::int8_t ao,
                  cl::sycl::buffer<std::uint8_t, 1> &b, std::int64_t ldb, std::uint8_t bo,
                  float beta, cl::sycl::buffer<std::int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<std::int32_t, 1> &co);

DLL_EXPORT void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                  std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                  cl::sycl::buffer<std::int8_t, 1> &a, std::int64_t lda, std::int8_t ao,
                  cl::sycl::buffer<std::int8_t, 1> &b, std::int64_t ldb, std::int8_t bo,
                  float beta, cl::sycl::buffer<std::int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<std::int32_t, 1> &co);

DLL_EXPORT void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                  std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                  cl::sycl::buffer<std::uint8_t, 1> &a, std::int64_t lda, std::uint8_t ao,
                  cl::sycl::buffer<std::int8_t, 1> &b, std::int64_t ldb, std::int8_t bo,
                  float beta, cl::sycl::buffer<std::int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<std::int32_t, 1> &co);

DLL_EXPORT void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                  std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                  cl::sycl::buffer<std::uint8_t, 1> &a, std::int64_t lda, std::uint8_t ao,
                  cl::sycl::buffer<std::uint8_t, 1> &b, std::int64_t ldb, std::uint8_t bo,
                  float beta, cl::sycl::buffer<std::int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<std::int32_t, 1> &co);

