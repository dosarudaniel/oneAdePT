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

#ifndef _MKL_RNG_DEVICE_MRG32K3A_IMPL_HPP_
#define _MKL_RNG_DEVICE_MRG32K3A_IMPL_HPP_

#include "oneapi/mkl/rng/device/detail/mrg32k3a_skip_ahead_matrix.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <std::int32_t VecSize = 1>
class mrg32k3a;

namespace detail {

// specialization of engine_state_device for mrg32k3a engine for generic DPC++ version
template <std::int32_t VecSize>
struct engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic> {
    std::uint32_t s[6];
};

// specialization of engine_state for mrg32k3a
template <std::int32_t VecSize>
union engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>
        generic_state;
};

namespace mrg32k3a_impl {

// MRG32k3a modules
//      MRG32K3A_M1 = 2^32 - 209
//      MRG32K3A_M2 = 2^32 - 22853
#define MRG32K3A_M1 4294967087
#define MRG32K3A_M2 4294944443

// MRG32k3a multipliers
#define MRG32K3A_A12  1403580
#define MRG32K3A_A13  4294156359
#define MRG32K3A_A21  527612
#define MRG32K3A_A23  4293573854
#define MRG32K3A_A13N 810728
#define MRG32K3A_A23N 1370589

// y = (x * (a ^ n)) % M
template <std::uint32_t M>
static inline void mv_mod(std::uint32_t a[3][3], std::uint32_t x[3], std::uint32_t y[3]) {
    std::uint64_t temp[3];
    for (int i = 0; i < 3; i++) {
        temp[i] = 0;
        for (int k = 0; k < 3; k++) {
            temp[i] += ((std::uint64_t)a[i][k] * (std::uint64_t)x[k]) % (std::uint64_t)M;
            temp[i] -= (temp[i] >= (std::uint64_t)M) * (std::uint64_t)M;
        }
    }
    for (int k = 0; k < 3; k++) {
        y[k] = (std::uint32_t)temp[k];
    }
    return;
}

// c = (a * b) % M
template <std::uint32_t M>
static inline void mm_mod(const std::uint32_t a[3][3], std::uint32_t b[3][3],
                          std::uint32_t c[3][3]) {
    std::uint64_t temp[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                temp[i][j] += ((std::uint64_t)a[i][k] * (std::uint64_t)b[k][j]) % (std::uint64_t)M;
                temp[i][j] -= (temp[i][j] >= (std::uint64_t)M) * (std::uint64_t)M;
            }
        }
    }
    c[0][0] = (std::uint32_t)temp[0][0];
    c[0][1] = (std::uint32_t)temp[0][1];
    c[0][2] = (std::uint32_t)temp[0][2];
    c[1][0] = (std::uint32_t)temp[1][0];
    c[1][1] = (std::uint32_t)temp[1][1];
    c[1][2] = (std::uint32_t)temp[1][2];
    c[2][0] = (std::uint32_t)temp[2][0];
    c[2][1] = (std::uint32_t)temp[2][1];
    c[2][2] = (std::uint32_t)temp[2][2];
}

// b = (a ^ n) % M
template <std::uint32_t M>
static inline void mm_pow_mod(std::uint32_t a[3][3], std::uint32_t b[3][3], std::uint64_t n) {
    b[0][0] = 1;
    b[0][1] = 0;
    b[0][2] = 0;
    b[1][0] = 0;
    b[1][1] = 1;
    b[1][2] = 0;
    b[2][0] = 0;
    b[2][1] = 0;
    b[2][2] = 1;
    if (n == 0) {
        return;
    }
    while (n) {
        if (n & 1) {
            mm_mod<M>(a, b, b);
        }
        n >>= 1;
        mm_mod<M>(a, a, a);
    }
}

template <std::uint32_t M>
static inline void mm_pow_mod_precomp(std::uint32_t a[3][3], std::uint64_t n,
                                      const std::uint64_t* skip_params,
                                      const std::uint32_t skip_ahead_matrix[193][3][3]) {
    std::uint32_t tmp;
    const std::uint32_t* skip_params_32 = reinterpret_cast<const std::uint32_t*>(skip_params);
    a[0][0] = 1;
    a[0][1] = 0;
    a[0][2] = 0;
    a[1][0] = 0;
    a[1][1] = 1;
    a[1][2] = 0;
    a[2][0] = 0;
    a[2][1] = 0;
    a[2][2] = 1;
    int i;
    for (int j = 0; j < 2 * n; j++) {
        tmp = skip_params_32[j];
        i = 0;
        while (tmp) {
            if (tmp & (1 << i)) {
                mm_mod<M>(skip_ahead_matrix[j * 32 + i], a, a);
                tmp &= ~(1 << i);
            }
            i++;
        }
    }
}

template <std::uint32_t M>
static inline void vec_pow_mod_precomp(std::uint32_t x[3], std::uint64_t n,
                                       const std::uint64_t* skip_params,
                                       const std::uint32_t skip_ahead_matrix[193][3][3]) {
    std::uint32_t temp[3][3];
    mm_pow_mod_precomp<M>(temp, n, skip_params, skip_ahead_matrix);
    mv_mod<M>(temp, x, x);
}

// x = (x * (b ^ n)) % M
template <std::uint32_t M>
static inline void vec_pow_mod(std::uint32_t x[3], std::uint32_t a[3], std::uint64_t n) {
    std::uint32_t b[3][3], c[3][3];
    b[0][0] = 0;
    b[0][1] = 1;
    b[0][2] = 0;
    b[1][0] = 0;
    b[1][1] = 0;
    b[1][2] = 1;
    b[2][0] = a[2];
    b[2][1] = a[1];
    b[2][2] = a[0];

    // C = (B ^ n) % m
    mm_pow_mod<M>(b, c, n);
    // x = (C * x) % m
    mv_mod<M>(c, x, x);
}

template <std::int32_t VecSize>
static inline void skip_ahead(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    std::uint64_t num_to_skip) {
    std::uint32_t a[3], x[3];
    a[0] = 0;
    a[1] = MRG32K3A_A12;
    a[2] = MRG32K3A_A13;
    x[0] = state.s[0];
    x[1] = state.s[1];
    x[2] = state.s[2];
    vec_pow_mod<MRG32K3A_M1>(x, a, num_to_skip);
    state.s[0] = x[0];
    state.s[1] = x[1];
    state.s[2] = x[2];

    a[0] = MRG32K3A_A21;
    a[1] = 0;
    a[2] = MRG32K3A_A23;
    x[0] = state.s[3];
    x[1] = state.s[4];
    x[2] = state.s[5];
    vec_pow_mod<MRG32K3A_M2>(x, a, num_to_skip);
    state.s[3] = x[0];
    state.s[4] = x[1];
    state.s[5] = x[2];
}

template <std::int32_t VecSize>
static inline void skip_ahead(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    std::uint64_t n, const std::uint64_t* num_to_skip_ptr) {
    if (n > 3) {
        n = 3;
#ifndef __SYCL_DEVICE_ONLY__
        throw oneapi::mkl::invalid_argument("rng", "mrg32k3a",
                                            "period is 2 ^ 191, skip on more than 2^192");
#endif
    }
    std::uint32_t x[3];
    x[0] = state.s[0];
    x[1] = state.s[1];
    x[2] = state.s[2];
    vec_pow_mod_precomp<MRG32K3A_M1>(x, n, num_to_skip_ptr, skip_ahead_table[0]);
    state.s[0] = x[0];
    state.s[1] = x[1];
    state.s[2] = x[2];

    x[0] = state.s[3];
    x[1] = state.s[4];
    x[2] = state.s[5];
    vec_pow_mod_precomp<MRG32K3A_M2>(x, n, num_to_skip_ptr, skip_ahead_table[1]);
    state.s[3] = x[0];
    state.s[4] = x[1];
    state.s[5] = x[2];
}

template <std::int32_t VecSize>
static inline void validate_seed(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state) {
    int i;
    for (i = 0; i < 3; i++) {
        if (state.s[i] >= MRG32K3A_M1) {
            state.s[i] -= MRG32K3A_M1;
        }
    }
    for (; i < 6; i++) {
        if (state.s[i] >= MRG32K3A_M2) {
            state.s[i] -= MRG32K3A_M2;
        }
    }

    if ((state.s[0]) == 0 && (state.s[1]) == 0 && (state.s[2]) == 0) {
        state.s[0] = 1;
    }
    if ((state.s[3]) == 0 && (state.s[4]) == 0 && (state.s[5]) == 0) {
        state.s[3] = 1;
    }
}

template <std::int32_t VecSize>
static inline void init(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    std::uint64_t n, const std::uint32_t* seed_ptr, std::uint64_t offset) {
    int i;
    if (n > 6) {
        n = 6;
    }
    for (i = 0; i < n; i++) {
        state.s[i] = seed_ptr[i];
    }
    for (; i < 6; i++) {
        state.s[i] = 1;
    }
    validate_seed(state);
    skip_ahead(state, offset);
}

template <std::int32_t VecSize>
static inline void init(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    std::uint64_t n, const std::uint32_t* seed_ptr, std::uint64_t n_offset,
    const std::uint64_t* offset_ptr) {
    int i;
    if (n > 6) {
        n = 6;
    }
    for (i = 0; i < n; i++) {
        state.s[i] = seed_ptr[i];
    }
    for (; i < 6; i++) {
        state.s[i] = 1;
    }
    validate_seed(state);
    mrg32k3a_impl::skip_ahead(state, n_offset, offset_ptr);
}

template <std::int32_t VecSize>
static inline void init(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    size_t id, const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>& accessor) {
    size_t num_elements_acc =
        sizeof(engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>>) / sizeof(std::uint32_t);
    state.s[0] = accessor[id * num_elements_acc];
    state.s[1] = accessor[id * num_elements_acc + 1];
    state.s[2] = accessor[id * num_elements_acc + 2];
    state.s[3] = accessor[id * num_elements_acc + 3];
    state.s[4] = accessor[id * num_elements_acc + 4];
    state.s[5] = accessor[id * num_elements_acc + 5];
}

template <std::int32_t VecSize>
static inline void store(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    size_t id, const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>& accessor) {
    size_t num_elements_acc =
        sizeof(engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>>) / sizeof(std::uint32_t);
    accessor[id * num_elements_acc] = state.s[0];
    accessor[id * num_elements_acc + 1] = state.s[1];
    accessor[id * num_elements_acc + 2] = state.s[2];
    accessor[id * num_elements_acc + 3] = state.s[3];
    accessor[id * num_elements_acc + 4] = state.s[4];
    accessor[id * num_elements_acc + 5] = state.s[5];
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;
    std::int64_t s11, s12, s13, s21, s22, s23;
    std::int64_t x, y;
    std::uint32_t tmp;
    std::int32_t i = 0;
    s13 = state.s[0];
    s12 = state.s[1];
    s11 = state.s[2];
    s23 = state.s[3];
    s22 = state.s[4];
    s21 = state.s[5];
    for (i = 0; i < num_elements; i++) {
        x = (MRG32K3A_A12 * s12 - MRG32K3A_A13N * s13) % MRG32K3A_M1;
        y = (MRG32K3A_A21 * s21 - MRG32K3A_A23N * s23) % MRG32K3A_M2;
        if (x < 0) {
            x += MRG32K3A_M1;
        }
        if (y < 0) {
            y += MRG32K3A_M2;
        }
        s13 = s12;
        s12 = s11;
        s11 = x;
        s23 = s22;
        s22 = s21;
        s21 = y;
        if (x <= y) {
            tmp = x + (MRG32K3A_M1 - y);
        }
        else {
            tmp = x - y;
        }
        res[i] = tmp;
    }
    state.s[0] = s13;
    state.s[1] = s12;
    state.s[2] = s11;
    state.s[3] = s23;
    state.s[4] = s22;
    state.s[5] = s21;
    return res;
}

template <std::int32_t VecSize>
static inline std::uint32_t generate_single(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state) {
    std::uint32_t res;
    std::int64_t s11, s12, s13, s21, s22, s23;
    std::int64_t x, y;
    std::uint32_t tmp;
    s13 = state.s[0];
    s12 = state.s[1];
    s11 = state.s[2];
    s23 = state.s[3];
    s22 = state.s[4];
    s21 = state.s[5];

    x = (MRG32K3A_A12 * s12 - MRG32K3A_A13N * s13) % MRG32K3A_M1;
    y = (MRG32K3A_A21 * s21 - MRG32K3A_A23N * s23) % MRG32K3A_M2;
    if (x < 0) {
        x += MRG32K3A_M1;
    }
    if (y < 0) {
        y += MRG32K3A_M2;
    }
    s13 = s12;
    s12 = s11;
    s11 = x;
    s23 = s22;
    s22 = s21;
    s21 = y;
    if (x <= y) {
        tmp = x + (MRG32K3A_M1 - y);
    }
    else {
        tmp = x - y;
    }
    res = tmp;

    state.s[0] = s13;
    state.s[1] = s12;
    state.s[2] = s11;
    state.s[3] = s23;
    state.s[4] = s22;
    state.s[5] = s21;
    return res;
}

} // namespace mrg32k3a_impl

// specialization for engine_accessor_base for mrg32k3a
template <std::int32_t VecSize>
class engine_accessor_base<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
public:
    engine_accessor_base(sycl::buffer<std::uint32_t, 1>& state_buf, sycl::handler& cgh)
            : states_accessor_(state_buf, cgh) {}

    oneapi::mkl::rng::device::mrg32k3a<VecSize> load(size_t id) const {
        oneapi::mkl::rng::device::mrg32k3a<VecSize> engine;
        mrg32k3a_impl::init(engine.state_.generic_state, id, states_accessor_);
        return engine;
    }

    void store(oneapi::mkl::rng::device::mrg32k3a<VecSize>& engine, size_t id) const {
        mrg32k3a_impl::store(engine.state_.generic_state, id, states_accessor_);
    }

protected:
    sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write> states_accessor_;
};

// specialization for engine_descriptor_base for mrg32k3a
template <std::int32_t VecSize>
class engine_descriptor_base<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
public:
    using engine_type = oneapi::mkl::rng::device::mrg32k3a<VecSize>;

    using accessor_type =
        oneapi::mkl::rng::device::engine_accessor<oneapi::mkl::rng::device::mrg32k3a<VecSize>>;

    engine_descriptor_base(sycl::queue& queue, sycl::range<1> range, std::uint32_t seed,
                           std::uint64_t offset)
            : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                             sizeof(std::uint32_t)) {
        queue.submit([&](sycl::handler& cgh) {
            accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel<engine_type>>
                (range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                oneapi::mkl::rng::device::mrg32k3a<VecSize> engine(seed, offset* id);
                states_accessor.store(engine, id);
            });
        });
    }

    template <typename InitEngineFunc>
    engine_descriptor_base(sycl::queue& queue, sycl::range<1> range, InitEngineFunc func)
            : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                             sizeof(std::uint32_t)) {
        queue.submit([&](sycl::handler& cgh) {
            accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel_ex<engine_type>>
                (range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                states_accessor.store(func(item), id);
            });
        });
    }

    accessor_type get_access(sycl::handler& cgh) {
        return accessor_type{ states_buffer_, cgh };
    }

protected:
    sycl::buffer<std::uint32_t, 1> states_buffer_;
};

template <std::int32_t VecSize>
class engine_base<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
protected:
    engine_base(std::uint32_t seed, std::uint64_t offset = 0) {
        mrg32k3a_impl::init(this->state_.generic_state, 1, &seed, offset);
    }

    engine_base(std::uint64_t n, const std::uint32_t* seed, std::uint64_t offset = 0) {
        mrg32k3a_impl::init(this->state_.generic_state, n, seed, offset);
    }

    engine_base(std::uint32_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        mrg32k3a_impl::init(this->state_.generic_state, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint32_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        mrg32k3a_impl::init(this->state_.generic_state, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;
        RealType c;

        c = (b - a) / (static_cast<RealType>(MRG32K3A_M1));

        res_uint = mrg32k3a_impl::generate(this->state_.generic_state);

        for (int i = 0; i < VecSize; i++) {
            res[i] = (RealType)(res_uint[i]) * c + a;
        }
        return res;
    }

    auto generate() -> typename std::conditional<VecSize == 1, std::uint32_t,
                                                 sycl::vec<std::uint32_t, VecSize>>::type {
        return mrg32k3a_impl::generate(this->state_.generic_state);
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;
        RealType c;

        c = (b - a) / (static_cast<RealType>(MRG32K3A_M1));

        res_uint = mrg32k3a_impl::generate_single(this->state_.generic_state);

        res = (RealType)(res_uint)*c + a;

        return res;
    }

    std::uint32_t generate_single() {
        return mrg32k3a_impl::generate_single(this->state_.generic_state);
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::mrg32k3a_impl::skip_ahead(this->state_.generic_state, num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        detail::mrg32k3a_impl::skip_ahead(this->state_.generic_state, num_to_skip.size(),
                                          num_to_skip.begin());
    }

    engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>> state_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_MRG32K3A_IMPL_HPP_
