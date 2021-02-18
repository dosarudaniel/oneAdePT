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

#ifndef _MKL_RNG_DEVICE_DISTRIBUTIONS_HPP_
#define _MKL_RNG_DEVICE_DISTRIBUTIONS_HPP_

#include <limits>

#include "oneapi/mkl/rng/device/detail/distribution_base.hpp"
#include "oneapi/mkl/rng/device/functions.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

// CONTINUOUS AND DISCRETE RANDOM NUMBER DISTRIBUTIONS

// Class template oneapi::mkl::rng::device::uniform
//
// Represents continuous and discrete uniform random number distribution
//
// Supported types:
//      float
//      double
//      std::int32_t
//
// Supported methods:
//      oneapi::mkl::rng::device::uniform_method::standard
//      oneapi::mkl::rng::device::uniform_method::accurate - for float and double types only
//
// Input arguments:
//      a - left bound. 0.0 by default
//      b - right bound. 1.0 by default (std::numeric_limits<std::int32_t>::max() for std::int32_t)

template <typename Type, typename Method>
class uniform : detail::distribution_base<uniform<Type, Method>> {
public:
    static_assert(std::is_same<Method, uniform_method::standard>::value ||
                      (std::is_same<Method, uniform_method::accurate>::value &&
                       !std::is_same<Type, std::int32_t>::value),
                  "oneMKL: rng/uniform: method is incorrect");

    static_assert(std::is_same<Type, float>::value || std::is_same<Type, double>::value ||
                      std::is_same<Type, std::int32_t>::value,
                  "oneMKL: rng/uniform: type is not supported");

    using method_type = Method;
    using result_type = Type;

    uniform()
            : detail::distribution_base<uniform<Type, Method>>(static_cast<Type>(0.0),
                                                               static_cast<Type>(1.0)) {}

    explicit uniform(Type a, Type b) : detail::distribution_base<uniform<Type, Method>>(a, b) {}

    Type a() const {
        return detail::distribution_base<uniform<Type, Method>>::a();
    }

    Type b() const {
        return detail::distribution_base<uniform<Type, Method>>::b();
    }

private:
    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;

    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

template <>
class uniform<std::int32_t, uniform_method::standard>
        : detail::distribution_base<uniform<std::int32_t, uniform_method::standard>> {
public:
    using method_type = uniform_method::standard;
    using result_type = std::int32_t;

    uniform()
            : detail::distribution_base<uniform<std::int32_t, uniform_method::standard>>(
                  0, std::numeric_limits<std::int32_t>::max()) {}

    explicit uniform(std::int32_t a, std::int32_t b)
            : detail::distribution_base<uniform<std::int32_t, uniform_method::standard>>(a, b) {}

    std::int32_t a() const {
        return detail::distribution_base<uniform<std::int32_t, uniform_method::standard>>::a();
    }

    std::int32_t b() const {
        return detail::distribution_base<uniform<std::int32_t, uniform_method::standard>>::b();
    }

private:
    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::gaussian
//
// Represents continuous normal random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::device::gaussian_method::box_muller2
//
// Input arguments:
//      mean   - mean. 0 by default
//      stddev - standard deviation. 1.0 by default
template <typename RealType, typename Method>
class gaussian : detail::distribution_base<gaussian<RealType, Method>> {
public:
    static_assert(std::is_same<Method, gaussian_method::box_muller2>::value,
                  "oneMKL: rng/gaussian: method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/gaussian: type is not supported");

    using method_type = Method;
    using result_type = RealType;

    gaussian()
            : detail::distribution_base<gaussian<RealType, Method>>(static_cast<RealType>(0.0),
                                                                    static_cast<RealType>(1.0)) {}

    explicit gaussian(RealType mean, RealType stddev)
            : detail::distribution_base<gaussian<RealType, Method>>(mean, stddev) {}

    RealType mean() const {
        return detail::distribution_base<gaussian<RealType, Method>>::mean();
    }

    RealType stddev() const {
        return detail::distribution_base<gaussian<RealType, Method>>::stddev();
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::lognormal
//
// Represents continuous lognormal random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::lognormal_method::box_muller2
//
// Input arguments:
//      m     - mean of the subject normal distribution. 0.0 by default
//      s     - standard deviation of the subject normal distribution. 1.0 by default
//      displ - displacement. 0.0 by default
//      scale - scalefactor. 1.0 by default
template <typename RealType, typename Method>
class lognormal : detail::distribution_base<lognormal<RealType, Method>> {
public:
    static_assert(std::is_same<Method, lognormal_method::box_muller2>::value,
                  "oneMKL: rng/lognormal: method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/lognormal: type is not supported");

    using method_type = Method;
    using result_type = RealType;

    lognormal()
            : detail::distribution_base<lognormal<RealType, Method>>(
                  static_cast<RealType>(0.0), static_cast<RealType>(1.0),
                  static_cast<RealType>(0.0), static_cast<RealType>(1.0)) {}

    explicit lognormal(RealType m, RealType s, RealType displ = static_cast<RealType>(0.0),
                       RealType scale = static_cast<RealType>(1.0))
            : detail::distribution_base<lognormal<RealType, Method>>(m, s, displ, scale) {}

    RealType m() const {
        return detail::distribution_base<lognormal<RealType, Method>>::m();
    }

    RealType s() const {
        return detail::distribution_base<lognormal<RealType, Method>>::s();
    }

    RealType displ() const {
        return detail::distribution_base<lognormal<RealType, Method>>::displ();
    }

    RealType scale() const {
        return detail::distribution_base<lognormal<RealType, Method>>::scale();
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::bits
//
// Represents bits of underlying random number engine
//
// Supported types:
//      std::uint32_t
//
template <typename UIntType>
class bits : detail::distribution_base<bits<UIntType>> {
public:
    static_assert(std::is_same<UIntType, std::uint32_t>::value,
                  "oneMKL: rng/bits: method is incorrect");
    using result_type = UIntType;

private:
    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;

    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_DISTRIBUTIONS_HPP_
