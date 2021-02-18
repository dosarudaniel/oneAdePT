/*******************************************************************************
* Copyright 2020 Intel Corporation.
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

#ifndef _MKL_RNG_DEVICE_FUNCTIONS_HPP_
#define _MKL_RNG_DEVICE_FUNCTIONS_HPP_

#include <CL/sycl.hpp>

#include "oneapi/mkl/rng/device/detail/distribution_base.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

// GENERATE FUNCTIONS

template <typename Distr, typename Engine>
auto generate(Distr& distr, Engine& engine) ->
    typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                              sycl::vec<typename Distr::result_type, Engine::vec_size>>::type {
    return distr.generate(engine);
}

template <typename Distr, typename Engine>
typename Distr::result_type generate_single(Distr& distr, Engine& engine) {
    static_assert(Engine::vec_size > 1,
                  "oneMKL: rng/generate_single: function works for engines with vec_size > 1");
    return distr.generate_single(engine);
}

// SERVICE FUNCTIONS

template <typename Engine>
void skip_ahead(Engine& engine, std::uint64_t num_to_skip) {
    engine.skip_ahead(num_to_skip);
}

template <typename Engine>
void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip) {
    engine.skip_ahead(num_to_skip);
}

} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_FUNCTIONS_HPP_
