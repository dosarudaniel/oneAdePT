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

#ifndef _MKL_RNG_DEVICE_ENGINES_HPP_
#define _MKL_RNG_DEVICE_ENGINES_HPP_

#include <limits>

#include "oneapi/mkl/rng/device/types.hpp"
#include "oneapi/mkl/rng/device/functions.hpp"
#include "oneapi/mkl/rng/device/detail/engine_base.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

// PSEUDO-RANDOM NUMBER HOST-SIDE ENGINE HELPERS

template <typename EngineType>
class engine_accessor : detail::engine_accessor_base<EngineType> {
public:
    EngineType load(size_t id) const {
        return detail::engine_accessor_base<EngineType>::load(id);
    }

    void store(EngineType engine, size_t id) const {
        detail::engine_accessor_base<EngineType>::store(engine, id);
    }

private:
    engine_accessor(sycl::buffer<std::uint32_t, 1>& state_buf, sycl::handler& cgh)
            : detail::engine_accessor_base<EngineType>(state_buf, cgh) {}
    friend detail::engine_descriptor_base<EngineType>;
};

template <typename EngineType = philox4x32x10<>>
class engine_descriptor : detail::engine_descriptor_base<EngineType> {};

template <std::int32_t VecSize>
class engine_descriptor<philox4x32x10<VecSize>>
        : detail::engine_descriptor_base<philox4x32x10<VecSize>> {
public:
    engine_descriptor(sycl::queue& queue, sycl::range<1> range, std::uint64_t seed,
                      std::uint64_t offset)
            : detail::engine_descriptor_base<philox4x32x10<VecSize>>(queue, range, seed, offset) {}

    template <typename InitEngineFunc>
    engine_descriptor(sycl::queue& queue, sycl::range<1> range, InitEngineFunc func)
            : detail::engine_descriptor_base<philox4x32x10<VecSize>>(queue, range, func) {}

    auto get_access(sycl::handler& cgh) {
        return detail::engine_descriptor_base<philox4x32x10<VecSize>>::get_access(cgh);
    }
};

template <std::int32_t VecSize>
class engine_descriptor<mrg32k3a<VecSize>> : detail::engine_descriptor_base<mrg32k3a<VecSize>> {
public:
    engine_descriptor(sycl::queue& queue, sycl::range<1> range, std::uint32_t seed,
                      std::uint64_t offset)
            : detail::engine_descriptor_base<mrg32k3a<VecSize>>(queue, range, seed, offset) {}

    template <typename InitEngineFunc>
    engine_descriptor(sycl::queue& queue, sycl::range<1> range, InitEngineFunc func)
            : detail::engine_descriptor_base<mrg32k3a<VecSize>>(queue, range, func) {}

    auto get_access(sycl::handler& cgh) {
        return detail::engine_descriptor_base<mrg32k3a<VecSize>>::get_access(cgh);
    }
};

// PSEUDO-RANDOM NUMBER DEVICE-SIDE ENGINES

// Class template oneapi::mkl::rng::device::philox4x32x10
//
// Represents Philox4x32-10 counter-based pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead

template <std::int32_t VecSize>
class philox4x32x10 : detail::engine_base<philox4x32x10<VecSize>> {
public:
    static constexpr std::uint64_t default_seed = 0;

    static constexpr std::int32_t vec_size = VecSize;

    philox4x32x10() : detail::engine_base<philox4x32x10<VecSize>>(default_seed) {}

    philox4x32x10(std::uint64_t seed, std::uint64_t offset = 0)
            : detail::engine_base<philox4x32x10<VecSize>>(seed, offset) {}

    philox4x32x10(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)
            : detail::engine_base<philox4x32x10<VecSize>>(seed.size(), seed.begin(), offset) {}

    philox4x32x10(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<philox4x32x10<VecSize>>(seed, offset.size(), offset.begin()) {}

    philox4x32x10(std::initializer_list<std::uint64_t> seed,
                  std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<philox4x32x10<VecSize>>(seed.size(), seed.begin(), offset.size(),
                                                          offset.begin()) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    friend class detail::engine_descriptor_base<philox4x32x10<VecSize>>;

    friend class detail::engine_accessor_base<philox4x32x10<VecSize>>;

    template <typename DistrType>
    friend class detail::distribution_base;
};

// Class oneapi::mkl::rng::device::mrg32k3a
//
// Represents the combined recurcive pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
template <std::int32_t VecSize>
class mrg32k3a : detail::engine_base<mrg32k3a<VecSize>> {
public:
    static constexpr std::uint32_t default_seed = 1;

    static constexpr std::int32_t vec_size = VecSize;

    mrg32k3a() : detail::engine_base<mrg32k3a<VecSize>>(default_seed) {}

    mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0)
            : detail::engine_base<mrg32k3a<VecSize>>(seed, offset) {}

    mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0)
            : detail::engine_base<mrg32k3a<VecSize>>(seed.size(), seed.begin(), offset) {}

    mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<mrg32k3a<VecSize>>(seed, offset.size(), offset.begin()) {}

    mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<mrg32k3a<VecSize>>(seed.size(), seed.begin(), offset.size(),
                                                     offset.begin()) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    friend class detail::engine_descriptor_base<mrg32k3a<VecSize>>;

    friend class detail::engine_accessor_base<mrg32k3a<VecSize>>;

    template <typename DistrType>
    friend class detail::distribution_base;
};

} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_ENGINES_HPP_
