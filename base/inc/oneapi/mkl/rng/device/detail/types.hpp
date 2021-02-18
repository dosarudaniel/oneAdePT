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

#ifndef _MKL_RNG_DETAIL_DETAIL_TYPES_HPP_
#define _MKL_RNG_DETAIL_DETAIL_TYPES_HPP_

#include <CL/sycl.hpp>

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <std::int32_t VecSize>
class philox4x32x10;

namespace detail {

// Type of device
namespace device_type {
struct generic {}; // currently only generic DPC++ version is supported
} // namespace device_type

// internal structure to specify state of engine for each device
template <typename EngineType, typename DeviceType>
struct engine_state_device {};

template <typename EngineType>
union engine_state {};

typedef struct {
    std::uint32_t hex[2];
} dp_struct_t;

typedef struct {
    std::uint32_t hex[1];
} sp_struct_t;

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DETAIL_DETAIL_TYPES_HPP_
