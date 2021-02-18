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

#ifndef _MKL_RNG_FUNCTIONS_HPP_
#define _MKL_RNG_FUNCTIONS_HPP_

#include "CL/sycl.hpp"

namespace oneapi {
namespace mkl {
namespace rng {

// GENERATE FUNCTIONS

// Function oneapi::mkl::rng::generate(). Buffer-based API
//
// Provides random numbers from a given engine with a given statistics
//
// Input parameters:
//      const Distr& distr              - distribution object
//      Engine& engine                   - engine object
//      std::int64_t n                   - number of random values to be generated
//
// Output parameters:
//      sycl::buffer<Distr::result_type, 1>& r - sycl::buffer to the output vector
template <typename Distr, typename Engine>
void generate(const Distr& distr, Engine& engine, std::int64_t n,
              sycl::buffer<typename Distr::result_type, 1>& r);

// Function oneapi::mkl::rng::generate(). USM-based API
//
// Provides random numbers from a given engine with a given statistics
//
// Input parameters:
//      const Distr& distr               - distribution object
//      Engine& engine                   - engine object
//      std::int64_t n                   - number of random values to be generated
//      const sycl::vector_class<sycl::event> &dependencies - list of events to wait for
//                  before starting computation, if any. If omitted, defaults to no dependencies
//
// Output parameters:
//      Distr::result_type* - pointer to the output vector
//
// Returns:
//      sycl::event - event for the submitted to the engine's queue task
template <typename Distr, typename Engine>
sycl::event generate(const Distr& distr, Engine& engine, std::int64_t n,
                     typename Distr::result_type* r,
                     const sycl::vector_class<sycl::event>& dependencies = {});

// SERVICE FUNCTIONS

// Function oneapi::mkl::rng::skip_ahead(). Common interface
//
// Proceeds state of engine using the skip-ahead method
//
// Input parameters:
//      Engine& engine             - engine object
//      const std::int64_t num_to_skip - number of skipped elements
template <typename Engine>
void skip_ahead(Engine& engine, std::uint64_t num_to_skip) {
    engine.skip_ahead(num_to_skip);
}

// Function oneapi::mkl::rng::skip_ahead(). Interface with partitioned number of skipped elements
//
// Proceeds state of engine using the skip-ahead method
//
// Input parameters:
//      Engine& engine                               - engine object
//      std::initializer_list<std::uint64_t> num_to_skip - number of skipped elements
template <typename Engine>
void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip) {
    engine.skip_ahead(num_to_skip);
}

// Function oneapi::mkl::rng::leapfrog()
//
// Proceeds state of engine using the leapfrog method
//
// Input parameters:
//      Engine& engine  - engine object
//      std::uint64_t idx    - index of the computational node
//      std::uint64_t stride - largest number of computational nodes, or stride
template <typename Engine>
void leapfrog(Engine& engine, std::uint64_t idx, std::uint64_t stride) {
    engine.leapfrog(idx, stride);
}

} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_FUNCTIONS_HPP_
