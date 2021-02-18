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

#ifndef _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
#define _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename Type>
class distribution_base<oneapi::mkl::rng::device::uniform<Type, uniform_method::standard>> {
public:
    distribution_base(Type a, Type b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, Type,
                                  sycl::vec<Type, EngineType::vec_size>>::type {
        return engine.generate(a_, b_);
    }

    template <typename EngineType>
    Type generate_single(EngineType& engine) {
        return engine.generate_single(a_, b_);
    }

    Type a_;
    Type b_;

    friend class distribution_base<
        oneapi::mkl::rng::device::gaussian<Type, gaussian_method::box_muller2>>;
};

// specialization for accurate method
template <typename Type>
class distribution_base<oneapi::mkl::rng::device::uniform<Type, uniform_method::accurate>> {
public:
    distribution_base(Type a, Type b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, Type,
                                  sycl::vec<Type, EngineType::vec_size>>::type {
        if constexpr (EngineType::vec_size == 1) {
            Type res = engine.generate(a_, b_);
            if (res < a_) {
                res = a_;
            }
            if (res > b_) {
                res = b_;
            }
            return res;
        }
        sycl::vec<Type, EngineType::vec_size> res = engine.generate(a_, b_);
        res = sycl::fmax(res, a_);
        res = sycl::fmin(res, b_);
        return res;
    }

    template <typename EngineType>
    Type generate_single(EngineType& engine) {
        Type res = engine.generate_single(a_, b_);
        if (res < a_) {
            res = a_;
        }
        if (res > b_) {
            res = b_;
        }
        return res;
    }

    Type a_;
    Type b_;
};

// specialization for std::int32_t
template <>
class distribution_base<oneapi::mkl::rng::device::uniform<std::int32_t>> {
public:
    distribution_base(std::int32_t a, std::int32_t b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
    }

    std::int32_t a() const {
        return a_;
    }

    std::int32_t b() const {
        return b_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, std::int32_t,
                                  sycl::vec<std::int32_t, EngineType::vec_size>>::type {
        if constexpr (EngineType::vec_size == 1) {
            float res_fp = engine.generate(static_cast<float>(a_), static_cast<float>(b_));
            res_fp = sycl::floor(res_fp);
            std::int32_t res_int = static_cast<int>(res_fp);
            return res_int;
        }

        sycl::vec<float, EngineType::vec_size> res_fp;
        res_fp = engine.generate(static_cast<float>(a_), static_cast<float>(b_));
        res_fp = sycl::floor(res_fp);
        auto res_int = res_fp.template convert<std::int32_t>();
        return res_int;
    }

    template <typename EngineType>
    std::int32_t generate_single(EngineType& engine) {
        float res_fp = engine.generate_single(static_cast<float>(a_), static_cast<float>(b_));
        res_fp = sycl::floor(res_fp);
        std::int32_t res_int = static_cast<int>(res_fp);
        return res_int;
    }

    std::int32_t a_;
    std::int32_t b_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
