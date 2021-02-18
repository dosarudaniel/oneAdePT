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

#ifndef _LAPACK_HPP__
#define _LAPACK_HPP__

//
// DPC++ MKL LAPACK API
//

#include <cstdint>
#include <complex>

#include <CL/sycl.hpp>

#include "oneapi/mkl/export.hpp"
#include "mkl_types.h"
#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/exceptions.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {

namespace internal
{
	// auxilary type aliases and forward declarations
	template <bool, typename T=void> struct enable_if;
	template <typename T> struct is_fp;
	template <typename T> struct is_rfp;
	template <typename T> struct is_cfp;
	template <typename fp> using is_floating_point         = typename enable_if<is_fp<fp>::value>::type*;
	template <typename fp> using is_real_floating_point    = typename enable_if<is_rfp<fp>::value>::type*;
	template <typename fp> using is_complex_floating_point = typename enable_if<is_cfp<fp>::value>::type*;
}

// potrf
DLL_EXPORT cl::sycl::event potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float  *a, std::int64_t lda, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a, std::int64_t lda, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potrf_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda);

// potrs
DLL_EXPORT cl::sycl::event potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, float  *a, std::int64_t lda, float  *b, std::int64_t ldb, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, double *a, std::int64_t lda, double *b, std::int64_t ldb, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *b, std::int64_t ldb, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::complex<double> *a, std::int64_t lda, std::complex<double> *b, std::int64_t ldb, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &b, std::int64_t ldb, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &b, std::int64_t ldb, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &b, std::int64_t ldb, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &b, std::int64_t ldb, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potrs_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t ldb);

// potri
DLL_EXPORT cl::sycl::event potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float  *a, std::int64_t lda, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a, std::int64_t lda, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potri(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potri_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda);

// gebrd: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t gebrd_scratchpad_size(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t lda);

// gebrd: USM API
DLL_EXPORT cl::sycl::event gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> *a, int64_t lda,
                                 float *d, float *e, std::complex<float> *tauq, std::complex<float> *taup,
                                 std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, double *a, int64_t lda, double *d,
                                 double *e, double *tauq, double *taup, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, float *a, int64_t lda, float *d,
                                 float *e, float *tauq, float *taup, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> *a, int64_t lda,
                                 double *d, double *e, std::complex<double> *tauq, std::complex<double> *taup,
                                 std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// gebrd: Buffer API
DLL_EXPORT void gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<std::complex<float>> &a,
                      int64_t lda, cl::sycl::buffer<float> &d, cl::sycl::buffer<float> &e,
                      cl::sycl::buffer<std::complex<float>> &tauq, cl::sycl::buffer<std::complex<float>> &taup,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<double> &d, cl::sycl::buffer<double> &e, cl::sycl::buffer<double> &tauq,
                      cl::sycl::buffer<double> &taup, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<float> &d, cl::sycl::buffer<float> &e, cl::sycl::buffer<float> &tauq,
                      cl::sycl::buffer<float> &taup, cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void gebrd(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<std::complex<double>> &a,
                      int64_t lda, cl::sycl::buffer<double> &d, cl::sycl::buffer<double> &e,
                      cl::sycl::buffer<std::complex<double>> &tauq, cl::sycl::buffer<std::complex<double>> &taup,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// geqrf: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t geqrf_scratchpad_size(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t lda);

// geqrf: USM API
DLL_EXPORT cl::sycl::event geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> *a, int64_t lda,
                                 std::complex<float> *tau, std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau,
                                 double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau,
                                 float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> *a, int64_t lda,
                                 std::complex<double> *tau, std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// geqrf: Buffer API
DLL_EXPORT void geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<std::complex<float>> &a,
                      int64_t lda, cl::sycl::buffer<std::complex<float>> &tau,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<float> &tau, cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void geqrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<std::complex<double>> &a,
                      int64_t lda, cl::sycl::buffer<std::complex<double>> &tau,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// gesvd: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t gesvd_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m,
                              int64_t n, int64_t lda, int64_t ldu, int64_t ldvt);

// gesvd: USM API
DLL_EXPORT cl::sycl::event gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m,
                                 int64_t n, double *a, int64_t lda, double *s, double *u, int64_t ldu, double *vt,
                                 int64_t ldvt, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m,
                                 int64_t n, float *a, int64_t lda, float *s, float *u, int64_t ldu, float *vt,
                                 int64_t ldvt, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// gesvd: Buffer API
DLL_EXPORT void gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                      cl::sycl::buffer<double> &a, int64_t lda, cl::sycl::buffer<double> &s,
                      cl::sycl::buffer<double> &u, int64_t ldu, cl::sycl::buffer<double> &vt, int64_t ldvt,
                      cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                      cl::sycl::buffer<float> &a, int64_t lda, cl::sycl::buffer<float> &s, cl::sycl::buffer<float> &u,
                      int64_t ldu, cl::sycl::buffer<float> &vt, int64_t ldvt, cl::sycl::buffer<float> &scratchpad,
                      int64_t scratchpad_size);

// gesvd: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t gesvd_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m,
                              int64_t n, int64_t lda, int64_t ldu, int64_t ldvt);

// gesvd: USM API
DLL_EXPORT cl::sycl::event gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m,
                                 int64_t n, std::complex<float> *a, int64_t lda, float *s, std::complex<float> *u,
                                 int64_t ldu, std::complex<float> *vt, int64_t ldvt, std::complex<float> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m,
                                 int64_t n, std::complex<double> *a, int64_t lda, double *s, std::complex<double> *u,
                                 int64_t ldu, std::complex<double> *vt, int64_t ldvt, std::complex<double> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// gesvd: Buffer API
DLL_EXPORT void gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                      cl::sycl::buffer<std::complex<float>> &a, int64_t lda, cl::sycl::buffer<float> &s,
                      cl::sycl::buffer<std::complex<float>> &u, int64_t ldu, cl::sycl::buffer<std::complex<float>> &vt,
                      int64_t ldvt, cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void gesvd(cl::sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                      cl::sycl::buffer<std::complex<double>> &a, int64_t lda, cl::sycl::buffer<double> &s,
                      cl::sycl::buffer<std::complex<double>> &u, int64_t ldu,
                      cl::sycl::buffer<std::complex<double>> &vt, int64_t ldvt,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// getrf: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t getrf_scratchpad_size(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t lda);

// getrf: USM API
DLL_EXPORT cl::sycl::event getrf(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> *a, int64_t lda,
                                 int64_t *ipiv, std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getrf(cl::sycl::queue &queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t *ipiv,
                                 double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getrf(cl::sycl::queue &queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t *ipiv,
                                 float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getrf(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> *a, int64_t lda,
                                 int64_t *ipiv, std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// getrf: Buffer API
DLL_EXPORT void getrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<std::complex<float>> &a,
                      int64_t lda, cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<std::complex<float>> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void getrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void getrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void getrf(cl::sycl::queue &queue, int64_t m, int64_t n, cl::sycl::buffer<std::complex<double>> &a,
                      int64_t lda, cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<std::complex<double>> &scratchpad,
                      int64_t scratchpad_size);

// getri: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t getri_scratchpad_size(cl::sycl::queue &queue, int64_t n, int64_t lda);

// getri: USM API
DLL_EXPORT cl::sycl::event getri(cl::sycl::queue &queue, int64_t n, std::complex<float> *a, int64_t lda, int64_t *ipiv,
                                 std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getri(cl::sycl::queue &queue, int64_t n, double *a, int64_t lda, int64_t *ipiv,
                                 double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getri(cl::sycl::queue &queue, int64_t n, float *a, int64_t lda, int64_t *ipiv,
                                 float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getri(cl::sycl::queue &queue, int64_t n, std::complex<double> *a, int64_t lda, int64_t *ipiv,
                                 std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// getri: Buffer API
DLL_EXPORT void getri(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>> &a, int64_t lda,
                      cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<std::complex<float>> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void getri(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void getri(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void getri(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>> &a, int64_t lda,
                      cl::sycl::buffer<int64_t> &ipiv, cl::sycl::buffer<std::complex<double>> &scratchpad,
                      int64_t scratchpad_size);

// getrs: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t getrs_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                              int64_t lda, int64_t ldb);

// getrs: USM API
DLL_EXPORT cl::sycl::event getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                                 std::complex<float> *a, int64_t lda, int64_t *ipiv, std::complex<float> *b,
                                 int64_t ldb, std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                                 double *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                                 float *a, int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                                 std::complex<double> *a, int64_t lda, int64_t *ipiv, std::complex<double> *b,
                                 int64_t ldb, std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// getrs: Buffer API
DLL_EXPORT void getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                      cl::sycl::buffer<std::complex<float>> &a, int64_t lda, cl::sycl::buffer<int64_t> &ipiv,
                      cl::sycl::buffer<std::complex<float>> &b, int64_t ldb,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                      cl::sycl::buffer<double> &a, int64_t lda, cl::sycl::buffer<int64_t> &ipiv,
                      cl::sycl::buffer<double> &b, int64_t ldb, cl::sycl::buffer<double> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                      cl::sycl::buffer<float> &a, int64_t lda, cl::sycl::buffer<int64_t> &ipiv,
                      cl::sycl::buffer<float> &b, int64_t ldb, cl::sycl::buffer<float> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void getrs(cl::sycl::queue &queue, oneapi::mkl::transpose trans, int64_t n, int64_t nrhs,
                      cl::sycl::buffer<std::complex<double>> &a, int64_t lda, cl::sycl::buffer<int64_t> &ipiv,
                      cl::sycl::buffer<std::complex<double>> &b, int64_t ldb,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// hetrf
DLL_EXPORT cl::sycl::event hetrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::int64_t *ipiv, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event hetrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::int64_t *ipiv, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void hetrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::int64_t> &ipiv, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void hetrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::int64_t> &ipiv, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t hetrf_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda);

// orgbr
DLL_EXPORT cl::sycl::event orgbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, float  *a, std::int64_t lda, float  *tau, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event orgbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, double *a, std::int64_t lda, double *tau, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void orgbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &tau, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void orgbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>
std::int64_t orgbr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::generate vect, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda);

// orgqr: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t orgqr_scratchpad_size(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, int64_t lda);

// orgqr: USM API
DLL_EXPORT cl::sycl::event orgqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda,
                                 double *tau, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event orgqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda,
                                 float *tau, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// orgqr: Buffer API
DLL_EXPORT void orgqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void orgqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<float> &tau, cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);

// ormqr: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t ormqr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, int64_t m,
                              int64_t n, int64_t k, int64_t lda, int64_t ldc);

// ormqr: USM API
DLL_EXPORT cl::sycl::event ormqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                 int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c,
                                 int64_t ldc, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event ormqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                 int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c,
                                 int64_t ldc, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// ormqr: Buffer API
DLL_EXPORT void ormqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, int64_t m,
                      int64_t n, int64_t k, cl::sycl::buffer<double> &a, int64_t lda, cl::sycl::buffer<double> &tau,
                      cl::sycl::buffer<double> &c, int64_t ldc, cl::sycl::buffer<double> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void ormqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, int64_t m,
                      int64_t n, int64_t k, cl::sycl::buffer<float> &a, int64_t lda, cl::sycl::buffer<float> &tau,
                      cl::sycl::buffer<float> &c, int64_t ldc, cl::sycl::buffer<float> &scratchpad,
                      int64_t scratchpad_size);

// ungbr
DLL_EXPORT cl::sycl::event ungbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *tau, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event ungbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> *a, std::int64_t lda, std::complex<double> *tau, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void ungbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &tau, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void ungbr(cl::sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t ungbr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::generate vect, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda);

// gerqf
DLL_EXPORT cl::sycl::event gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float  *a, std::int64_t lda, float  *tau, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda, double *tau, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *tau, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::complex<double> *tau, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &tau, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &tau, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void gerqf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t gerqf_scratchpad_size(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t lda);

// ormrq
DLL_EXPORT cl::sycl::event ormrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, float  *a, std::int64_t lda, float  *tau, float  *c, std::int64_t ldc, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event ormrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, double *a, std::int64_t lda, double *tau, double *c, std::int64_t ldc, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void ormrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &tau, cl::sycl::buffer<float>  &c, std::int64_t ldc, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void ormrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &c, std::int64_t ldc, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>
std::int64_t ormrq_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc);

// unmrq
DLL_EXPORT cl::sycl::event unmrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *tau, std::complex<float>  *c, std::int64_t ldc, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event unmrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> *a, std::int64_t lda, std::complex<double> *tau, std::complex<double> *c, std::int64_t ldc, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void unmrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &tau, cl::sycl::buffer<std::complex<float>>  &c, std::int64_t ldc, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void unmrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &c, std::int64_t ldc, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t unmrq_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc);

// heev: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t heev_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                             int64_t lda);

// heev: USM API
DLL_EXPORT cl::sycl::event heev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                                std::complex<float> *a, int64_t lda, float *w, std::complex<float> *scratchpad,
                                int64_t scratchpad_size,
                                const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event heev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                                std::complex<double> *a, int64_t lda, double *w, std::complex<double> *scratchpad,
                                int64_t scratchpad_size,
                                const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// heev: Buffer API
DLL_EXPORT void heev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                     cl::sycl::buffer<std::complex<float>> &a, int64_t lda, cl::sycl::buffer<float> &w,
                     cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void heev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                     cl::sycl::buffer<std::complex<double>> &a, int64_t lda, cl::sycl::buffer<double> &w,
                     cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// heevd: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t heevd_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                              int64_t lda);

// heevd: USM API
DLL_EXPORT cl::sycl::event heevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                                 std::complex<float> *a, int64_t lda, float *w, std::complex<float> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event heevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                                 std::complex<double> *a, int64_t lda, double *w, std::complex<double> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// heevd: Buffer API
DLL_EXPORT void heevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<std::complex<float>> &a, int64_t lda, cl::sycl::buffer<float> &w,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void heevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<std::complex<double>> &a, int64_t lda, cl::sycl::buffer<double> &w,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// heevx: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t heevx_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                              oneapi::mkl::uplo uplo, int64_t n, int64_t lda, int64_t il, int64_t iu, int64_t ldz);

// heevx: USM API
DLL_EXPORT cl::sycl::event heevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                                 oneapi::mkl::uplo uplo, int64_t n, std::complex<float> *a, int64_t lda, float vl,
                                 float vu, int64_t il, int64_t iu, float abstol, int64_t *m, float *w,
                                 std::complex<float> *z, int64_t ldz, std::complex<float> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event heevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                                 oneapi::mkl::uplo uplo, int64_t n, std::complex<double> *a, int64_t lda, double vl,
                                 double vu, int64_t il, int64_t iu, double abstol, int64_t *m, double *w,
                                 std::complex<double> *z, int64_t ldz, std::complex<double> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// heevx: Buffer API
DLL_EXPORT void heevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<std::complex<float>> &a, int64_t lda,
                      float vl, float vu, int64_t il, int64_t iu, float abstol, cl::sycl::buffer<int64_t> &m,
                      cl::sycl::buffer<float> &w, cl::sycl::buffer<std::complex<float>> &z, int64_t ldz,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void heevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<std::complex<double>> &a, int64_t lda,
                      double vl, double vu, int64_t il, int64_t iu, double abstol, cl::sycl::buffer<int64_t> &m,
                      cl::sycl::buffer<double> &w, cl::sycl::buffer<std::complex<double>> &z, int64_t ldz,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// hegvd: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t hegvd_scratchpad_size(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                              int64_t n, int64_t lda, int64_t ldb);

// hegvd: USM API
DLL_EXPORT cl::sycl::event hegvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                 int64_t n, std::complex<float> *a, int64_t lda, std::complex<float> *b, int64_t ldb,
                                 float *w, std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event hegvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                 int64_t n, std::complex<double> *a, int64_t lda, std::complex<double> *b, int64_t ldb,
                                 double *w, std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// hegvd: Buffer API
DLL_EXPORT void hegvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<std::complex<float>> &a, int64_t lda, cl::sycl::buffer<std::complex<float>> &b,
                      int64_t ldb, cl::sycl::buffer<float> &w, cl::sycl::buffer<std::complex<float>> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void hegvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<std::complex<double>> &a, int64_t lda, cl::sycl::buffer<std::complex<double>> &b,
                      int64_t ldb, cl::sycl::buffer<double> &w, cl::sycl::buffer<std::complex<double>> &scratchpad,
                      int64_t scratchpad_size);

// hegvx: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t hegvx_scratchpad_size(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                              oneapi::mkl::uplo uplo, int64_t n, int64_t lda, int64_t ldb, int64_t il, int64_t iu,
                              int64_t ldz);

// hegvx: USM API
DLL_EXPORT cl::sycl::event hegvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz,
                                 oneapi::mkl::rangev range, oneapi::mkl::uplo uplo, int64_t n, std::complex<float> *a,
                                 int64_t lda, std::complex<float> *b, int64_t ldb, float vl, float vu, int64_t il,
                                 int64_t iu, float abstol, int64_t *m, float *w, std::complex<float> *z, int64_t ldz,
                                 std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event hegvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz,
                                 oneapi::mkl::rangev range, oneapi::mkl::uplo uplo, int64_t n, std::complex<double> *a,
                                 int64_t lda, std::complex<double> *b, int64_t ldb, double vl, double vu, int64_t il,
                                 int64_t iu, double abstol, int64_t *m, double *w, std::complex<double> *z, int64_t ldz,
                                 std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// hegvx: Buffer API
DLL_EXPORT void hegvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<std::complex<float>> &a, int64_t lda,
                      cl::sycl::buffer<std::complex<float>> &b, int64_t ldb, float vl, float vu, int64_t il, int64_t iu,
                      float abstol, cl::sycl::buffer<int64_t> &m, cl::sycl::buffer<float> &w,
                      cl::sycl::buffer<std::complex<float>> &z, int64_t ldz,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void hegvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<std::complex<double>> &a, int64_t lda,
                      cl::sycl::buffer<std::complex<double>> &b, int64_t ldb, double vl, double vu, int64_t il,
                      int64_t iu, double abstol, cl::sycl::buffer<int64_t> &m, cl::sycl::buffer<double> &w,
                      cl::sycl::buffer<std::complex<double>> &z, int64_t ldz,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// hetrd: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t hetrd_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, int64_t lda);

// hetrd: USM API
DLL_EXPORT cl::sycl::event hetrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, std::complex<float> *a,
                                 int64_t lda, float *d, float *e, std::complex<float> *tau,
                                 std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event hetrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, std::complex<double> *a,
                                 int64_t lda, double *d, double *e, std::complex<double> *tau,
                                 std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// hetrd: Buffer API
DLL_EXPORT void hetrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<std::complex<float>> &a, int64_t lda, cl::sycl::buffer<float> &d,
                      cl::sycl::buffer<float> &e, cl::sycl::buffer<std::complex<float>> &tau,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void hetrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<std::complex<double>> &a, int64_t lda, cl::sycl::buffer<double> &d,
                      cl::sycl::buffer<double> &e, cl::sycl::buffer<std::complex<double>> &tau,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// sytrf
DLL_EXPORT cl::sycl::event sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float  *a, std::int64_t lda, std::int64_t *ipiv, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a, std::int64_t lda, std::int64_t *ipiv, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::int64_t *ipiv, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::int64_t *ipiv, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<std::int64_t> &ipiv, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<std::int64_t> &ipiv, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::int64_t> &ipiv, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void sytrf(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::int64_t> &ipiv, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t sytrf_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda);

// orgtr
DLL_EXPORT cl::sycl::event orgtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float  *a, std::int64_t lda, float  *tau, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event orgtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a, std::int64_t lda, double *tau, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void orgtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &tau, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void orgtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>
std::int64_t orgtr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda);

// ungtr
DLL_EXPORT cl::sycl::event ungtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *tau, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event ungtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::complex<double> *tau, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void ungtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &tau, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void ungtr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t ungtr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda);

// ormtr
DLL_EXPORT cl::sycl::event ormtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, float  *a, std::int64_t lda, float  *tau, float  *c, std::int64_t ldc, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event ormtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, double *a, std::int64_t lda, double *tau, double *c, std::int64_t ldc, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void ormtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, cl::sycl::buffer<float>  &tau, cl::sycl::buffer<float>  &c, std::int64_t ldc, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void ormtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &c, std::int64_t ldc, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>
std::int64_t ormtr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldc);

// unmtr
DLL_EXPORT cl::sycl::event unmtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::complex<float>  *tau, std::complex<float>  *c, std::int64_t ldc, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event unmtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::complex<double> *tau, std::complex<double> *c, std::int64_t ldc, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void unmtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, cl::sycl::buffer<std::complex<float>>  &tau, cl::sycl::buffer<std::complex<float>>  &c, std::int64_t ldc, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void unmtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &c, std::int64_t ldc, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t unmtr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldc);

// steqr: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t steqr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, int64_t ldz);

// steqr: USM API
DLL_EXPORT cl::sycl::event steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, float *d, float *e,
                                 std::complex<float> *z, int64_t ldz, std::complex<float> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, double *d, double *e,
                                 double *z, int64_t ldz, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, float *d, float *e,
                                 float *z, int64_t ldz, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, double *d, double *e,
                                 std::complex<double> *z, int64_t ldz, std::complex<double> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// steqr: Buffer API
DLL_EXPORT void steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, cl::sycl::buffer<float> &d,
                      cl::sycl::buffer<float> &e, cl::sycl::buffer<std::complex<float>> &z, int64_t ldz,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, cl::sycl::buffer<double> &d,
                      cl::sycl::buffer<double> &e, cl::sycl::buffer<double> &z, int64_t ldz,
                      cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, cl::sycl::buffer<float> &d,
                      cl::sycl::buffer<float> &e, cl::sycl::buffer<float> &z, int64_t ldz,
                      cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void steqr(cl::sycl::queue &queue, oneapi::mkl::compz compz, int64_t n, cl::sycl::buffer<double> &d,
                      cl::sycl::buffer<double> &e, cl::sycl::buffer<std::complex<double>> &z, int64_t ldz,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// syev: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t syev_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                             int64_t lda);

// syev: USM API
DLL_EXPORT cl::sycl::event syev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                                double *a, int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size,
                                const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event syev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                                float *a, int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size,
                                const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// syev: Buffer API
DLL_EXPORT void syev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                     cl::sycl::buffer<double> &a, int64_t lda, cl::sycl::buffer<double> &w,
                     cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void syev(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::uplo uplo, int64_t n,
                     cl::sycl::buffer<float> &a, int64_t lda, cl::sycl::buffer<float> &w,
                     cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);

// syevd: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t syevd_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                              int64_t lda);

// syevd: USM API
DLL_EXPORT cl::sycl::event syevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                                 double *a, int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event syevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                                 float *a, int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// syevd: Buffer API
DLL_EXPORT void syevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<double> &a, int64_t lda, cl::sycl::buffer<double> &w,
                      cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void syevd(cl::sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<float> &a, int64_t lda, cl::sycl::buffer<float> &w,
                      cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);

// syevx: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t syevx_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                              oneapi::mkl::uplo uplo, int64_t n, int64_t lda, int64_t il, int64_t iu, int64_t ldz);

// syevx: USM API
DLL_EXPORT cl::sycl::event syevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                                 oneapi::mkl::uplo uplo, int64_t n, double *a, int64_t lda, double vl, double vu,
                                 int64_t il, int64_t iu, double abstol, int64_t *m, double *w, double *z, int64_t ldz,
                                 double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event syevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                                 oneapi::mkl::uplo uplo, int64_t n, float *a, int64_t lda, float vl, float vu,
                                 int64_t il, int64_t iu, float abstol, int64_t *m, float *w, float *z, int64_t ldz,
                                 float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// syevx: Buffer API
DLL_EXPORT void syevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<double> &a, int64_t lda, double vl, double vu,
                      int64_t il, int64_t iu, double abstol, cl::sycl::buffer<int64_t> &m, cl::sycl::buffer<double> &w,
                      cl::sycl::buffer<double> &z, int64_t ldz, cl::sycl::buffer<double> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void syevx(cl::sycl::queue &queue, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<float> &a, int64_t lda, float vl, float vu,
                      int64_t il, int64_t iu, float abstol, cl::sycl::buffer<int64_t> &m, cl::sycl::buffer<float> &w,
                      cl::sycl::buffer<float> &z, int64_t ldz, cl::sycl::buffer<float> &scratchpad,
                      int64_t scratchpad_size);

// sygvd: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t sygvd_scratchpad_size(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                              int64_t n, int64_t lda, int64_t ldb);

// sygvd: USM API
DLL_EXPORT cl::sycl::event sygvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                 int64_t n, double *a, int64_t lda, double *b, int64_t ldb, double *w,
                                 double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event sygvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                 int64_t n, float *a, int64_t lda, float *b, int64_t ldb, float *w, float *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// sygvd: Buffer API
DLL_EXPORT void sygvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<double> &a, int64_t lda, cl::sycl::buffer<double> &b, int64_t ldb,
                      cl::sycl::buffer<double> &w, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void sygvd(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
                      cl::sycl::buffer<float> &a, int64_t lda, cl::sycl::buffer<float> &b, int64_t ldb,
                      cl::sycl::buffer<float> &w, cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);

// sygvx: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t sygvx_scratchpad_size(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                              oneapi::mkl::uplo uplo, int64_t n, int64_t lda, int64_t ldb, int64_t il, int64_t iu,
                              int64_t ldz);

// sygvx: USM API
DLL_EXPORT cl::sycl::event sygvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz,
                                 oneapi::mkl::rangev range, oneapi::mkl::uplo uplo, int64_t n, double *a, int64_t lda,
                                 double *b, int64_t ldb, double vl, double vu, int64_t il, int64_t iu, double abstol,
                                 int64_t *m, double *w, double *z, int64_t ldz, double *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event sygvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz,
                                 oneapi::mkl::rangev range, oneapi::mkl::uplo uplo, int64_t n, float *a, int64_t lda,
                                 float *b, int64_t ldb, float vl, float vu, int64_t il, int64_t iu, float abstol,
                                 int64_t *m, float *w, float *z, int64_t ldz, float *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// sygvx: Buffer API
DLL_EXPORT void sygvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<double> &b, int64_t ldb, double vl, double vu, int64_t il, int64_t iu,
                      double abstol, cl::sycl::buffer<int64_t> &m, cl::sycl::buffer<double> &w,
                      cl::sycl::buffer<double> &z, int64_t ldz, cl::sycl::buffer<double> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void sygvx(cl::sycl::queue &queue, int64_t itype, oneapi::mkl::compz jobz, oneapi::mkl::rangev range,
                      oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<float> &b, int64_t ldb, float vl, float vu, int64_t il, int64_t iu, float abstol,
                      cl::sycl::buffer<int64_t> &m, cl::sycl::buffer<float> &w, cl::sycl::buffer<float> &z, int64_t ldz,
                      cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);

// sytrd: scratchpad size query
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
int64_t sytrd_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, int64_t lda);

// sytrd: USM API
DLL_EXPORT cl::sycl::event sytrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, double *a, int64_t lda,
                                 double *d, double *e, double *tau, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event sytrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, float *a, int64_t lda,
                                 float *d, float *e, float *tau, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// sytrd: Buffer API
DLL_EXPORT void sytrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<double> &a,
                      int64_t lda, cl::sycl::buffer<double> &d, cl::sycl::buffer<double> &e,
                      cl::sycl::buffer<double> &tau, cl::sycl::buffer<double> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void sytrd(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, int64_t n, cl::sycl::buffer<float> &a,
                      int64_t lda, cl::sycl::buffer<float> &d, cl::sycl::buffer<float> &e, cl::sycl::buffer<float> &tau,
                      cl::sycl::buffer<float> &scratchpad, int64_t scratchpad_size);

// trtrs: scratchpad size query
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
int64_t trtrs_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                              oneapi::mkl::diag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);

// trtrs: USM API
DLL_EXPORT cl::sycl::event trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag diag, int64_t n, int64_t nrhs, std::complex<float> *a, int64_t lda,
                                 std::complex<float> *b, int64_t ldb, std::complex<float> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag diag, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b,
                                 int64_t ldb, double *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag diag, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b,
                                 int64_t ldb, float *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag diag, int64_t n, int64_t nrhs, std::complex<double> *a, int64_t lda,
                                 std::complex<double> *b, int64_t ldb, std::complex<double> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// trtrs: Buffer API
DLL_EXPORT void trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                      oneapi::mkl::diag diag, int64_t n, int64_t nrhs, cl::sycl::buffer<std::complex<float>> &a,
                      int64_t lda, cl::sycl::buffer<std::complex<float>> &b, int64_t ldb,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                      oneapi::mkl::diag diag, int64_t n, int64_t nrhs, cl::sycl::buffer<double> &a, int64_t lda,
                      cl::sycl::buffer<double> &b, int64_t ldb, cl::sycl::buffer<double> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                      oneapi::mkl::diag diag, int64_t n, int64_t nrhs, cl::sycl::buffer<float> &a, int64_t lda,
                      cl::sycl::buffer<float> &b, int64_t ldb, cl::sycl::buffer<float> &scratchpad,
                      int64_t scratchpad_size);
DLL_EXPORT void trtrs(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                      oneapi::mkl::diag diag, int64_t n, int64_t nrhs, cl::sycl::buffer<std::complex<double>> &a,
                      int64_t lda, cl::sycl::buffer<std::complex<double>> &b, int64_t ldb,
                      cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

// ungqr: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t ungqr_scratchpad_size(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, int64_t lda);

// ungqr: USM API
DLL_EXPORT cl::sycl::event ungqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, std::complex<float> *a,
                                 int64_t lda, std::complex<float> *tau, std::complex<float> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event ungqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, std::complex<double> *a,
                                 int64_t lda, std::complex<double> *tau, std::complex<double> *scratchpad,
                                 int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// ungqr: Buffer API
DLL_EXPORT void ungqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k, cl::sycl::buffer<std::complex<float>> &a,
                      int64_t lda, cl::sycl::buffer<std::complex<float>> &tau,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void ungqr(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t k,
                      cl::sycl::buffer<std::complex<double>> &a, int64_t lda,
                      cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &scratchpad,
                      int64_t scratchpad_size);

// unmqr: scratchpad size query
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
int64_t unmqr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, int64_t m,
                              int64_t n, int64_t k, int64_t lda, int64_t ldc);

// unmqr: USM API
DLL_EXPORT cl::sycl::event unmqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                 int64_t m, int64_t n, int64_t k, std::complex<float> *a, int64_t lda,
                                 std::complex<float> *tau, std::complex<float> *c, int64_t ldc,
                                 std::complex<float> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});
DLL_EXPORT cl::sycl::event unmqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                 int64_t m, int64_t n, int64_t k, std::complex<double> *a, int64_t lda,
                                 std::complex<double> *tau, std::complex<double> *c, int64_t ldc,
                                 std::complex<double> *scratchpad, int64_t scratchpad_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &event_list = {});

// unmqr: Buffer API
DLL_EXPORT void unmqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, int64_t m,
                      int64_t n, int64_t k, cl::sycl::buffer<std::complex<float>> &a, int64_t lda,
                      cl::sycl::buffer<std::complex<float>> &tau, cl::sycl::buffer<std::complex<float>> &c, int64_t ldc,
                      cl::sycl::buffer<std::complex<float>> &scratchpad, int64_t scratchpad_size);
DLL_EXPORT void unmqr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, int64_t m,
                      int64_t n, int64_t k, cl::sycl::buffer<std::complex<double>> &a, int64_t lda,
                      cl::sycl::buffer<std::complex<double>> &tau, cl::sycl::buffer<std::complex<double>> &c,
                      int64_t ldc, cl::sycl::buffer<std::complex<double>> &scratchpad, int64_t scratchpad_size);

//
// DPC++ MKL LAPACK batch group API
//

DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, float  **a, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, double **a, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::complex<float>  **a, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::complex<double> **a, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, float  **a, std::int64_t *lda, float  **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, double **a, std::int64_t *lda, double **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, std::complex<float>  **a, std::int64_t *lda, std::complex<float>  **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, std::complex<double> **a, std::int64_t *lda, std::complex<double> **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, float  **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, double **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::complex<float>  **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::complex<double> **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs, float  **a, std::int64_t *lda, std::int64_t **ipiv, float  **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs, double **a, std::int64_t *lda, std::int64_t **ipiv, double **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs, std::complex<float>  **a, std::int64_t *lda, std::int64_t **ipiv, std::complex<float>  **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs, std::complex<double> **a, std::int64_t *lda, std::int64_t **ipiv, std::complex<double> **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t *n, float  **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t *n, double **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float>  **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, float  **a, std::int64_t *lda, float  **tau, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, double **a, std::int64_t *lda, double **tau, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::complex<float>  **a, std::int64_t *lda, std::complex<float>  **tau, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::complex<double> **a, std::int64_t *lda, std::complex<double> **tau, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event orgqr_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, float  **a, std::int64_t *lda, float  **tau, std::int64_t group_count, std::int64_t *group_sizes, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event orgqr_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, double **a, std::int64_t *lda, double **tau, std::int64_t group_count, std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

DLL_EXPORT cl::sycl::event ungqr_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float>  **a, std::int64_t *lda, std::complex<float>  **tau, std::int64_t group_count, std::int64_t *group_sizes, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event ungqr_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> **a, std::int64_t *lda, std::complex<double> **tau, std::int64_t group_count, std::int64_t *group_sizes, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});

template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potrf_batch_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potrs_batch_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, std::int64_t *lda, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t getrf_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t getrs_batch_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs, std::int64_t *lda, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t getri_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *n, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t geqrf_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>
std::int64_t orgqr_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t ungqr_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes);

//
// DPC++ MKL LAPACK batch stride API
//

DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrf_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, float  *a, std::int64_t lda, std::int64_t stride_a, float  *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, double *a, std::int64_t lda, std::int64_t stride_a, double *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::complex<float>  *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float>  &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<double> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::complex<float>>  &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void potrs_batch(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::complex<double>> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, float  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, float  *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, double *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, double *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::complex<float>  *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, cl::sycl::buffer<float>  &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, cl::sycl::buffer<double> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, cl::sycl::buffer<std::complex<float>>  &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getrs_batch(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, cl::sycl::buffer<std::complex<double>> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,  cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float  *a, std::int64_t lda, std::int64_t stride_a, float  *tau, std::int64_t stride_tau, std::int64_t batch_size, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda, std::int64_t stride_a, double *tau, std::int64_t stride_tau, std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::complex<float>  *tau, std::int64_t stride_tau, std::int64_t batch_size, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::complex<double> *tau, std::int64_t stride_tau, std::int64_t batch_size, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float>  &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<double> &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::complex<float>>  &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::complex<double>> &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event orgqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, float  *a, std::int64_t lda, std::int64_t stride_a, float  *tau, std::int64_t stride_tau, std::int64_t batch_size, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event orgqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, double *a, std::int64_t lda, std::int64_t stride_a, double *tau, std::int64_t stride_tau, std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void orgqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float>  &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void orgqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<double> &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event ungqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::complex<float>  *tau, std::int64_t stride_tau, std::int64_t batch_size, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event ungqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::complex<double> *tau, std::int64_t stride_tau, std::int64_t batch_size, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void ungqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::complex<float>>  &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void ungqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::complex<double>> &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t n, float  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, float  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t n, double *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<float>  *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, std::complex<float>  *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, std::complex<double> *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {});
DLL_EXPORT void getri_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<float>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getri_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getri_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>>  &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<std::complex<float>>  &scratchpad, std::int64_t scratchpad_size);
DLL_EXPORT void getri_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size);

template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potrf_batch_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t potrs_batch_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t getrf_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t getrs_batch_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t geqrf_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>
std::int64_t orgqr_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>
std::int64_t ungqr_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size);
template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>
std::int64_t getri_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t n, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size);

class DLL_EXPORT exception
{
public:
    exception(mkl::exception *_ex, std::int64_t info, std::int64_t detail = 0) : _info(info), _detail(detail), _ex(_ex) {}
    std::int64_t info()   const { return _info; }
    std::int64_t detail() const { return _detail; }
    const char*  what()   const { return _ex->what(); }
private:
    std::int64_t   _info;
    std::int64_t   _detail;
    mkl::exception *_ex;
};

class DLL_EXPORT computation_error : public oneapi::mkl::computation_error, public oneapi::mkl::lapack::exception
{
public:
    computation_error(const std::string &function, const std::string &info, std::int64_t code)
        : oneapi::mkl::computation_error("LAPACK", function, info), oneapi::mkl::lapack::exception(this, code) {}
    using oneapi::mkl::computation_error::what;
};

class DLL_EXPORT batch_error : public oneapi::mkl::batch_error, public oneapi::mkl::lapack::exception
{
public:
    batch_error(const std::string &function, const std::string &info, std::int64_t num_errors, cl::sycl::vector_class<std::int64_t> ids = {}, cl::sycl::vector_class<std::exception_ptr> exceptions = {})
            : oneapi::mkl::batch_error("LAPACK", function, info), oneapi::mkl::lapack::exception(this, num_errors), _ids(ids), _exceptions(exceptions) {}
    using oneapi::mkl::batch_error::what;
    const cl::sycl::vector_class<std::int64_t>& ids() const { return _ids; }
    const cl::sycl::vector_class<std::exception_ptr>& exceptions() const { return _exceptions; }
private:
    cl::sycl::vector_class<std::int64_t> _ids;
    cl::sycl::vector_class<std::exception_ptr> _exceptions;
};

class DLL_EXPORT invalid_argument : public oneapi::mkl::invalid_argument, public oneapi::mkl::lapack::exception
{
public:
    invalid_argument(const std::string &function, const std::string &info, std::int64_t arg_position = 0, std::int64_t detail = 0)
        : oneapi::mkl::invalid_argument("LAPACK", function, info), oneapi::mkl::lapack::exception(this, arg_position, detail) {}
    using oneapi::mkl::invalid_argument::what;
};

namespace internal
{
    // auxilary typechecking templates
    template<typename T>
    struct enable_if<true,T> { using type = T; };

    template<> struct is_fp<float>                { static constexpr bool value{true}; };
    template<> struct is_fp<double>               { static constexpr bool value{true}; };
    template<> struct is_fp<std::complex<float>>  { static constexpr bool value{true}; };
    template<> struct is_fp<std::complex<double>> { static constexpr bool value{true}; };

    template<> struct is_rfp<float>  { static constexpr bool value{true}; };
    template<> struct is_rfp<double> { static constexpr bool value{true}; };

    template<> struct is_cfp<std::complex<float>>  { static constexpr bool value{true}; };
    template<> struct is_cfp<std::complex<double>> { static constexpr bool value{true}; };
}

} //namespace lapack
} //namespace mkl
} // namespace oneapi


#endif  // _LAPACK_HPP__
