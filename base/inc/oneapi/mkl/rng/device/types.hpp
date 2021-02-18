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

#ifndef _MKL_RNG_DEVICE_TYPES_HPP_
#define _MKL_RNG_DEVICE_TYPES_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

// METHODS FOR DISTRIBUTIONS

namespace uniform_method {
struct standard {};
struct accurate {};
using by_default = standard;
} // namespace uniform_method

namespace gaussian_method {
struct box_muller2 {};
using by_default = box_muller2;
} // namespace gaussian_method

namespace lognormal_method {
struct box_muller2 {};
using by_default = box_muller2;
} // namespace lognormal_method

} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_TYPES_HPP_
