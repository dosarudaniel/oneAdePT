/*******************************************************************************
* Copyright 2019-2020 Intel Corporation.
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

#ifndef _SPBLAS_HPP_
#define _SPBLAS_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>
#include <stdexcept>

#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

class not_initialized_exception : public std::runtime_error
{
public:
    not_initialized_exception() : std::runtime_error("SPARSE_STATUS_NOT_INITIALIZED") {}
};

class allocation_failed_exception : public std::runtime_error
{
public:
    allocation_failed_exception() : std::runtime_error("SPARSE_STATUS_ALLOC_FAILED") {}
};

class invalid_value_exception : public std::runtime_error
{
public:
    invalid_value_exception() : std::runtime_error("SPARSE_STATUS_INVALID_VALUE") {}
};

class execution_failed_exception : public std::runtime_error
{
public:
    execution_failed_exception() : std::runtime_error("SPARSE_STATUS_EXECUTION_FAILED") {}
};

class internal_error_exception : public std::runtime_error
{
public:
    internal_error_exception() : std::runtime_error("SPARSE_STATUS_INTERNAL_ERROR") {}
};

class not_supported_exception : public std::runtime_error
{
public:
    not_supported_exception() : std::runtime_error("SPARSE_STATUS_NOT_SUPPORTED") {}
};

enum class property : char {
    symmetric = 0,
    sorted    = 1,
};

struct matrix_handle;
typedef struct matrix_handle *matrix_handle_t;

void init_matrix_handle(matrix_handle_t *handle);

void release_matrix_handle(matrix_handle_t *handle,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

void set_matrix_property(matrix_handle_t handle, property property_value);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int32_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int32_t, 1> &col_ind,
                  cl::sycl::buffer<float, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int64_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int64_t, 1> &col_ind,
                  cl::sycl::buffer<float, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int32_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int32_t, 1> &col_ind,
                  cl::sycl::buffer<double, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int64_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int64_t, 1> &col_ind,
                  cl::sycl::buffer<double, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int32_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int32_t, 1> &col_ind,
                  cl::sycl::buffer<std::complex<float>, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int64_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int64_t, 1> &col_ind,
                  cl::sycl::buffer<std::complex<float>, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int32_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int32_t, 1> &col_ind,
                  cl::sycl::buffer<std::complex<double>, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  cl::sycl::buffer<std::int64_t, 1> &row_ptr,
                  cl::sycl::buffer<std::int64_t, 1> &col_ind,
                  cl::sycl::buffer<std::complex<double>, 1> &val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  std::int32_t *row_ptr,
                  std::int32_t *col_ind,
                  float *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  std::int64_t *row_ptr,
                  std::int64_t *col_ind,
                  float *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  std::int32_t *row_ptr,
                  std::int32_t *col_ind,
                  double *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  std::int64_t *row_ptr,
                  std::int64_t *col_ind,
                  double *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  std::int32_t *row_ptr,
                  std::int32_t *col_ind,
                  std::complex<float> *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  std::int64_t *row_ptr,
                  std::int64_t *col_ind,
                  std::complex<float> *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int32_t num_rows,
                  const std::int32_t num_cols,
                  index_base index,
                  std::int32_t *row_ptr,
                  std::int32_t *col_ind,
                  std::complex<double> *val);

void set_csr_data(matrix_handle_t handle,
                  const std::int64_t num_rows,
                  const std::int64_t num_cols,
                  index_base index,
                  std::int64_t *row_ptr,
                  std::int64_t *col_ind,
                  std::complex<double> *val);

void make_transpose(cl::sycl::queue &queue,
                    transpose transpose_flag,
                    matrix_handle_t handle,
                    matrix_handle_t trans_handle);

void make_symmetric(cl::sycl::queue &queue,
                    uplo uplo_flag,
                    matrix_handle_t handle,
                    matrix_handle_t sym_handle);

void optimize_gemv(cl::sycl::queue &queue, transpose transpose_flag, matrix_handle_t handle);

cl::sycl::event optimize_gemv(cl::sycl::queue &queue,
                              transpose transpose_flag,
                              matrix_handle_t handle,
                              const cl::sycl::vector_class<cl::sycl::event> &dependencies);

void optimize_trmv(cl::sycl::queue &queue,
                   uplo uplo_flag,
                   transpose transpose_flag,
                   diag diag_val,
                   matrix_handle_t handle);

cl::sycl::event optimize_trmv(cl::sycl::queue &queue,
                              uplo uplo_flag,
                              transpose transpose_flag,
                              diag diag_val,
                              matrix_handle_t handle,
                              const cl::sycl::vector_class<cl::sycl::event> &dependencies);

void optimize_trsv(cl::sycl::queue &queue,
                   uplo uplo_flag,
                   transpose transpose_flag,
                   diag diag_val,
                   matrix_handle_t handle);

cl::sycl::event optimize_trsv(cl::sycl::queue &queue,
                              uplo uplo_flag,
                              transpose transpose_flag,
                              diag diag_val,
                              matrix_handle_t handle,
                              const cl::sycl::vector_class<cl::sycl::event> &dependencies);

void gemv(cl::sycl::queue &queue,
          transpose transpose_flag,
          const float alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<float, 1> &x,
          const float beta,
          cl::sycl::buffer<float, 1> &y);

void gemv(cl::sycl::queue &queue,
          transpose transpose_flag,
          const double alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<double, 1> &x,
          const double beta,
          cl::sycl::buffer<double, 1> &y);

void gemv(cl::sycl::queue &queue,
          transpose transpose_flag,
          const std::complex<float> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<float>, 1> &x,
          const std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y);

void gemv(cl::sycl::queue &queue,
          transpose transpose_flag,
          const std::complex<double> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<double>, 1> &x,
          const std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y);

cl::sycl::event gemv(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const float alpha,
                     matrix_handle_t handle,
                     const float *x,
                     const float beta,
                     float *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const double alpha,
                     matrix_handle_t handle,
                     const double *x,
                     const double beta,
                     double *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     const std::complex<float> *x,
                     const std::complex<float> beta,
                     std::complex<float> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     const std::complex<double> *x,
                     const std::complex<double> beta,
                     std::complex<double> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

void gemvdot(cl::sycl::queue &queue,
             transpose transpose_flag,
             const float alpha,
             matrix_handle_t handle,
             cl::sycl::buffer<float, 1> &x,
             const float beta,
             cl::sycl::buffer<float, 1> &y,
             cl::sycl::buffer<float, 1> &d);

void gemvdot(cl::sycl::queue &queue,
             transpose transpose_flag,
             const double alpha,
             matrix_handle_t handle,
             cl::sycl::buffer<double, 1> &x,
             const double beta,
             cl::sycl::buffer<double, 1> &y,
             cl::sycl::buffer<double, 1> &d);

void gemvdot(cl::sycl::queue &queue,
             transpose transpose_flag,
             const std::complex<float> alpha,
             matrix_handle_t handle,
             cl::sycl::buffer<std::complex<float>, 1> &x,
             const std::complex<float> beta,
             cl::sycl::buffer<std::complex<float>, 1> &y,
             cl::sycl::buffer<std::complex<float>, 1> &d);

void gemvdot(cl::sycl::queue &queue,
             transpose transpose_flag,
             const std::complex<double> alpha,
             matrix_handle_t handle,
             cl::sycl::buffer<std::complex<double>, 1> &x,
             const std::complex<double> beta,
             cl::sycl::buffer<std::complex<double>, 1> &y,
             cl::sycl::buffer<std::complex<double>, 1> &d);

cl::sycl::event gemvdot(cl::sycl::queue &queue,
                        transpose transpose_flag,
                        const float alpha,
                        matrix_handle_t handle,
                        float *x,
                        const float beta,
                        float *y,
                        float *d,
                        const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemvdot(cl::sycl::queue &queue,
                        transpose transpose_flag,
                        const double alpha,
                        matrix_handle_t handle,
                        double *x,
                        const double beta,
                        double *y,
                        double *d,
                        const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemvdot(cl::sycl::queue &queue,
                        transpose transpose_flag,
                        const std::complex<float> alpha,
                        matrix_handle_t handle,
                        std::complex<float> *x,
                        const std::complex<float> beta,
                        std::complex<float> *y,
                        std::complex<float> *d,
                        const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemvdot(cl::sycl::queue &queue,
                        transpose transpose_flag,
                        const std::complex<double> alpha,
                        matrix_handle_t handle,
                        std::complex<double> *x,
                        const std::complex<double> beta,
                        std::complex<double> *y,
                        std::complex<double> *d,
                        const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

void symv(cl::sycl::queue &queue,
          uplo uplo_flag,
          const float alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<float, 1> &x,
          const float beta,
          cl::sycl::buffer<float, 1> &y);

void symv(cl::sycl::queue &queue,
          uplo uplo_flag,
          const double alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<double, 1> &x,
          const double beta,
          cl::sycl::buffer<double, 1> &y);

void symv(cl::sycl::queue &queue,
          uplo uplo_flag,
          const std::complex<float> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<float>, 1> &x,
          const std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y);

void symv(cl::sycl::queue &queue,
          uplo uplo_flag,
          const std::complex<double> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<double>, 1> &x,
          const std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y);

cl::sycl::event symv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     const float alpha,
                     matrix_handle_t handle,
                     float *x,
                     const float beta,
                     float *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event symv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     const double alpha,
                     matrix_handle_t handle,
                     double *x,
                     const double beta,
                     double *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event symv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     std::complex<float> *x,
                     const std::complex<float> beta,
                     std::complex<float> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event symv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     std::complex<double> *x,
                     const std::complex<double> beta,
                     std::complex<double> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

void trmv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          const float alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<float, 1> &x,
          const float beta,
          cl::sycl::buffer<float, 1> &y);

void trmv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          const double alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<double, 1> &x,
          const double beta,
          cl::sycl::buffer<double, 1> &y);

void trmv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          const std::complex<float> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<float>, 1> &x,
          const std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y);

void trmv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          const std::complex<double> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<double>, 1> &x,
          const std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y);

cl::sycl::event trmv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     const float alpha,
                     matrix_handle_t handle,
                     float *x,
                     const float beta,
                     float *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     const double alpha,
                     matrix_handle_t handle,
                     double *x,
                     const double beta,
                     double *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     std::complex<float> *x,
                     const std::complex<float> beta,
                     std::complex<float> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     std::complex<double> *x,
                     const std::complex<double> beta,
                     std::complex<double> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

void trsv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          matrix_handle_t handle,
          cl::sycl::buffer<float, 1> &x,
          cl::sycl::buffer<float, 1> &y);

void trsv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          matrix_handle_t handle,
          cl::sycl::buffer<double, 1> &x,
          cl::sycl::buffer<double, 1> &y);

void trsv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<float>, 1> &x,
          cl::sycl::buffer<std::complex<float>, 1> &y);

void trsv(cl::sycl::queue &queue,
          uplo uplo_flag,
          transpose transpose_flag,
          diag diag_val,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<double>, 1> &x,
          cl::sycl::buffer<std::complex<double>, 1> &y);

cl::sycl::event trsv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     matrix_handle_t handle,
                     float *x,
                     float *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     matrix_handle_t handle,
                     double *x,
                     double *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     matrix_handle_t handle,
                     std::complex<float> *x,
                     std::complex<float> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_flag,
                     matrix_handle_t handle,
                     std::complex<double> *x,
                     std::complex<double> *y,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

void gemm(cl::sycl::queue &queue,
          transpose transpose_flag,
          const float alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<float, 1> &b,
          const std::int64_t columns,
          const std::int64_t ldb,
          const float beta,
          cl::sycl::buffer<float, 1> &c,
          const std::int64_t ldc);

void gemm(cl::sycl::queue &queue,
          transpose transpose_flag,
          const double alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<double, 1> &b,
          const std::int64_t columns,
          const std::int64_t ldb,
          const double beta,
          cl::sycl::buffer<double, 1> &c,
          const std::int64_t ldc);

void gemm(cl::sycl::queue &queue,
          transpose transpose_flag,
          const std::complex<float> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<float>, 1> &b,
          const std::int64_t columns,
          const std::int64_t ldb,
          const std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c,
          const std::int64_t ldc);

void gemm(cl::sycl::queue &queue,
          transpose transpose_flag,
          const std::complex<double> alpha,
          matrix_handle_t handle,
          cl::sycl::buffer<std::complex<double>, 1> &b,
          const std::int64_t columns,
          const std::int64_t ldb,
          const std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c,
          const std::int64_t ldc);

cl::sycl::event gemm(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const float alpha,
                     matrix_handle_t handle,
                     float *b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const float beta,
                     float *c,
                     const std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const double alpha,
                     matrix_handle_t handle,
                     double *b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const double beta,
                     double *c,
                     const std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     std::complex<float> *b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const std::complex<float> beta,
                     std::complex<float> *c,
                     const std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue,
                     transpose transpose_flag,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     std::complex<double> *b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const std::complex<double> beta,
                     std::complex<double> *c,
                     const std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

} /* namespace oneapi::mkl::sparse */
} /* namespace mkl */
} // namespace oneapi

#endif /* _SPBLAS_HPP_ */
