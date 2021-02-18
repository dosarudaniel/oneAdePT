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

#ifndef _MKL_RNG_DISTRIBUTION_BASE_HPP_
#define _MKL_RNG_DISTRIBUTION_BASE_HPP_

#include <CL/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/device/types.hpp"
#include "oneapi/mkl/rng/device/detail/types.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename DistrType>
class distribution_base {};

namespace distr_common {

static const dp_struct_t distr_dp_table[2]{
    { 0x54442D18, 0x401921FB }, // Pi * 2
    { 0x446D71C3, 0xC0874385 }, // ln(0.494065e-323) = -744.440072
};

static const sp_struct_t distr_sp_table[2]{
    { 0x40C90FDB }, // Pi * 2
    { 0xC2CE8ED0 }, // ln(0.14012984e-44) = -103.278929
};

template <typename RealType = float>
inline RealType ln_callback() {
    return ((const RealType*)(distr_sp_table))[1];
}

template <>
inline double ln_callback<double>() {
    return ((const double*)(distr_dp_table))[1];
}

template <typename RealType = float>
inline RealType pi2() {
    return ((const RealType*)(distr_sp_table))[0];
}

template <>
inline double pi2<double>() {
    return ((const double*)(distr_dp_table))[0];
}

} // namespace distr_common
} // namespace detail

// declarations of distribution classes
template <typename Type = float, typename Method = uniform_method::by_default>
class uniform;

template <typename RealType = float, typename Method = gaussian_method::by_default>
class gaussian;

template <typename RealType = float, typename Method = lognormal_method::by_default>
class lognormal;

template <typename UIntType = std::uint32_t>
class bits;

} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#include "oneapi/mkl/rng/device/detail/uniform_impl.hpp"
#include "oneapi/mkl/rng/device/detail/gaussian_impl.hpp"
#include "oneapi/mkl/rng/device/detail/lognormal_impl.hpp"
#include "oneapi/mkl/rng/device/detail/bits_impl.hpp"

#endif // _MKL_RNG_DISTRIBUTION_BASE_HPP_
