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

#ifndef _MKL_RNG_DEVICE_GAUSSIAN_IMPL_HPP_
#define _MKL_RNG_DEVICE_GAUSSIAN_IMPL_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename RealType>
class distribution_base<
    oneapi::mkl::rng::device::gaussian<RealType, gaussian_method::box_muller2>> {
public:
    distribution_base(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
        flag_ = false;
#ifndef __SYCL_DEVICE_ONLY__
        if (stddev <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        RealType u1, u2, u1_transformed;

        if constexpr (EngineType::vec_size == 1) {
            RealType res;
            if (!flag_) {
                u1 = uniform_.generate(engine);
                u2 = uniform_.generate(engine);
                u1_transformed = (u1 == static_cast<RealType>(0.0))
                                     ? distr_common::ln_callback<RealType>()
                                     : sycl::log(u1);
                u1_transformed = sycl::sqrt(static_cast<RealType>(-2.0) * u1_transformed);
                res = (u1_transformed * sycl::sin(distr_common::pi2<RealType>() * u2)) * stddev_ +
                      mean_;
                u1_transformed_ = u1_transformed;
                u2_ = u2;
                flag_ = true;
                return res;
            }
            res = (u1_transformed_ * sycl::cos(distr_common::pi2<RealType>() * u2_)) * stddev_ +
                  mean_;
            flag_ = false;
            return res;
        }
        else {
            sycl::vec<RealType, EngineType::vec_size> res;
            if (!flag_) {
                std::int32_t tail = EngineType::vec_size % 2;
                auto uniform_res = uniform_.generate(engine);
                for (std::int32_t i = 0; i < EngineType::vec_size - tail; i += 2) {
                    u1 = uniform_res[i];
                    u2 = uniform_res[i + 1];
                    u1_transformed = (u1 == static_cast<RealType>(0.0))
                                         ? distr_common::ln_callback<RealType>()
                                         : sycl::log(u1);
                    u1_transformed = sycl::sqrt(static_cast<RealType>(-2.0) * u1_transformed);
                    res[i] =
                        (u1_transformed * sycl::sin(distr_common::pi2<RealType>() * u2)) * stddev_ +
                        mean_;
                    res[i + 1] =
                        (u1_transformed * sycl::cos(distr_common::pi2<RealType>() * u2)) * stddev_ +
                        mean_;
                }
                if (tail) {
                    u1 = uniform_res[EngineType::vec_size - 1];
                    u2 = uniform_.generate_single(engine);
                    u1_transformed = (u1 == static_cast<RealType>(0.0))
                                         ? distr_common::ln_callback<RealType>()
                                         : sycl::log(u1);
                    u1_transformed = sycl::sqrt(static_cast<RealType>(-2.0) * u1_transformed);
                    res[EngineType::vec_size - 1] =
                        (u1_transformed * sycl::sin(distr_common::pi2<RealType>() * u2)) * stddev_ +
                        mean_;
                    u1_transformed_ = u1_transformed;
                    u2_ = u2;
                    flag_ = true;
                }
                return res;
            }

            res[0] = (u1_transformed_ * sycl::cos(distr_common::pi2<RealType>() * u2_)) * stddev_ +
                     mean_;
            flag_ = false;
            std::int32_t tail = (EngineType::vec_size - 1) % 2;
            for (std::int32_t i = 1; i < EngineType::vec_size - tail; i += 2) {
                u1 = uniform_.generate_single(engine);
                u2 = uniform_.generate_single(engine);
                u1_transformed = (u1 == static_cast<RealType>(0.0))
                                     ? distr_common::ln_callback<RealType>()
                                     : sycl::log(u1);
                u1_transformed = sycl::sqrt(static_cast<RealType>(-2.0) * u1_transformed);
                res[i] =
                    (u1_transformed * sycl::sin(distr_common::pi2<RealType>() * u2)) * stddev_ +
                    mean_;
                res[i + 1] =
                    (u1_transformed * sycl::cos(distr_common::pi2<RealType>() * u2)) * stddev_ +
                    mean_;
            }
            if (tail) {
                u1 = uniform_.generate_single(engine);
                u2 = uniform_.generate_single(engine);
                u1_transformed = (u1 == static_cast<RealType>(0.0))
                                     ? distr_common::ln_callback<RealType>()
                                     : sycl::log(u1);
                u1_transformed = sycl::sqrt(static_cast<RealType>(-2.0) * u1_transformed);
                res[EngineType::vec_size - 1] =
                    (u1_transformed * sycl::sin(distr_common::pi2<RealType>() * u2)) * stddev_ +
                    mean_;
                u1_transformed_ = u1_transformed;
                u2_ = u2;
                flag_ = true;
            }
            return res;
        }
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        RealType u1, u2, u1_transformed;
        RealType res;
        if (!flag_) {
            u1 = uniform_.generate_single(engine);
            u2 = uniform_.generate_single(engine);
            u1_transformed = (u1 == static_cast<RealType>(0.0))
                                 ? distr_common::ln_callback<RealType>()
                                 : sycl::log(u1);
            u1_transformed = sycl::sqrt(static_cast<RealType>(-2.0) * u1_transformed);
            res =
                (u1_transformed * sycl::sin(distr_common::pi2<RealType>() * u2)) * stddev_ + mean_;
            u1_transformed_ = u1_transformed;
            u2_ = u2;
            flag_ = true;
            return res;
        }
        res = (u1_transformed_ * sycl::cos(distr_common::pi2<RealType>() * u2_)) * stddev_ + mean_;
        flag_ = false;
        return res;
    }

    distribution_base<oneapi::mkl::rng::device::uniform<RealType>> uniform_ = {
        static_cast<RealType>(0.0), static_cast<RealType>(1.0)
    };
    RealType mean_;
    RealType stddev_;
    bool flag_ = false;
    RealType u1_transformed_;
    RealType u2_;

    friend class distribution_base<
        oneapi::mkl::rng::device::lognormal<RealType, lognormal_method::box_muller2>>;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_GAUSSIAN_IMPL_HPP_
