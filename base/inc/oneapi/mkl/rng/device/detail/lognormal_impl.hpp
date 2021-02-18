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

#ifndef _MKL_RNG_DEVICE_LOGNORMAL_IMPL_HPP_
#define _MKL_RNG_DEVICE_LOGNORMAL_IMPL_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename RealType, typename Method>
class distribution_base<oneapi::mkl::rng::device::lognormal<RealType, Method>> {
public:
    distribution_base(RealType m, RealType s, RealType displ, RealType scale)
            : gaussian_(m, s),
              displ_(displ),
              scale_(scale) {
#ifndef __SYCL_DEVICE_ONLY__
        if (scale <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "lognormal", "scale <= 0");
        }
#endif
    }

    RealType m() const {
        return gaussian_.mean();
    }

    RealType s() const {
        return gaussian_.stddev();
    }

    RealType displ() const {
        return displ_;
    }

    RealType scale() const {
        return scale_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        auto res = gaussian_.generate(engine);
        return sycl::exp(res) * scale_ + displ_;
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        RealType res = gaussian_.generate_single(engine);
        return sycl::exp(res) * scale_ + displ_;
    }

    distribution_base<oneapi::mkl::rng::device::gaussian<RealType, gaussian_method::box_muller2>>
        gaussian_;
    RealType displ_;
    RealType scale_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_LIGNORMAL_IMPL_HPP_
