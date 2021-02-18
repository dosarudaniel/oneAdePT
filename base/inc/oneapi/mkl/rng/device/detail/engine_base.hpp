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

#ifndef _MKL_RNG_DEVICE_ENGINE_BASE_HPP_
#define _MKL_RNG_DEVICE_ENGINE_BASE_HPP_

#include <CL/sycl.hpp>

#include "oneapi/mkl/rng/device/detail/types.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <typename EngineType>
class engine_accessor;

namespace detail {

template <typename EngineType>
class engine_base {
protected:
    template <typename RealType>
    auto generate(RealType a, RealType b) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type;

    auto generate() ->
        typename std::conditional<EngineType::vec_size == 1, std::uint32_t,
                                  sycl::vec<std::uint32_t, EngineType::vec_size>>::type;

    engine_state<EngineType> state_;
};

template <typename EngineType>
class engine_accessor_base {};

template <typename EngineType>
class init_kernel {};

template <typename EngineType>
class init_kernel_ex {};

template <typename EngineType>
class engine_descriptor_base {};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#include "oneapi/mkl/rng/device/detail/philox4x32x10_impl.hpp"
#include "oneapi/mkl/rng/device/detail/mrg32k3a_impl.hpp"

#endif // _MKL_RNG_DEVICE_ENGINE_BASE_HPP_
