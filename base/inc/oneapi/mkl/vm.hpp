/* file: oneapi/mkl/vm.hpp */
/*******************************************************************************
* Copyright 2019-2020 Intel Corporation.
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

#ifndef _ONEAPI_MKL_VM_HPP
#define _ONEAPI_MKL_VM_HPP 1

#include <cstdint>
#include <complex>
#include <exception>
#include <CL/sycl.hpp>

#ifdef MKL_BUILD_DLL
#define __MKL_VM_EXPORT     __declspec(dllexport)
#define __MKL_VM_EXPORT_CPP __declspec(dllexport)
#else
#define __MKL_VM_EXPORT extern
#define __MKL_VM_EXPORT_CPP
#endif

namespace oneapi {
namespace mkl {
namespace vm {

namespace enums {
    enum class mode : std::uint32_t {
        not_defined = 0x0,

        la = 0x1,
        ha = 0x2,
        ep = 0x3,

        global_status_report = 0x100,
    };

    enum class status : std::uint32_t {
        not_defined = 0x0,

        success = 0x0,

        errdom = 0x1,
        sing = 0x2,
        overflow = 0x4,
        underflow = 0x8,

        accuracy_warning = 0x80,
        fix_all = 0xff,
    };

    template <typename T>
    struct bits_enabled { static constexpr bool enabled = false; };

    template <>
    struct bits_enabled<mode> { static constexpr bool enabled = true; };

    template <>
    struct bits_enabled<status> { static constexpr bool enabled = true; };


    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, T>::type
    operator |(T lhs, T rhs) {
        auto r = static_cast<typename std::underlying_type_t<T>>(lhs) | static_cast<typename std::underlying_type_t<T>>(rhs);
        return static_cast<T>(r);
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, T>::type &
    operator |=(T & lhs, T rhs) {
        auto r = static_cast<typename std::underlying_type_t<T>>(lhs) | static_cast<typename std::underlying_type_t<T>>(rhs);
        lhs = static_cast<T>(r);
        return lhs;
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, T>::type
    operator &(T lhs, T rhs) {
        auto r = static_cast<typename std::underlying_type_t<T>>(lhs) & static_cast<typename std::underlying_type_t<T>>(rhs);
        return static_cast<T>(r);
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, T>::type
    operator &=(T & lhs, T rhs) {
        auto r = static_cast<typename std::underlying_type_t<T>>(lhs) & static_cast<typename std::underlying_type_t<T>>(rhs);
        lhs = static_cast<T>(r);
        return lhs;
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, T>::type
    operator ^(T lhs, T rhs) {
        auto r = static_cast<typename std::underlying_type_t<T>>(lhs) ^ static_cast<typename std::underlying_type_t<T>>(rhs);
        return static_cast<T>(r);
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, T>::type
    operator ^=(T & lhs, T rhs) {
        auto r = static_cast<typename std::underlying_type_t<T>>(lhs) ^ static_cast<typename std::underlying_type_t<T>>(rhs);
        lhs = static_cast<T>(r);
        return lhs;
    }


    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, bool>::type
    operator !(T v) { return (0 == static_cast<typename std::underlying_type_t<T>>(v)); }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, bool>::type
    has_any(T v, T mask) {
        auto r = static_cast<typename std::underlying_type_t<T>>(v) & static_cast<typename std::underlying_type_t<T>>(mask);
        return (0 != r);
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, bool>::type
    has_all(T v, T mask) {
        auto r = static_cast<typename std::underlying_type_t<T>>(v) & static_cast<typename std::underlying_type_t<T>>(mask);
        return (static_cast<typename std::underlying_type_t<T>>(mask) == r);
    }

    template <typename T>
    typename std::enable_if<bits_enabled<T>::enabled, bool>::type
    has_only(T v, T mask) {
        auto r = static_cast<typename std::underlying_type_t<T>>(v) ^ static_cast<typename std::underlying_type_t<T>>(mask);
        return (0 == r);
    }

} // namespace enums

    using enums::mode;
    using enums::status;

namespace detail {
    using std::int64_t;
    namespace one_vm = oneapi::mkl::vm;

    template <typename T>
    struct __MKL_VM_EXPORT_CPP error_handler {
        bool enabled_;
        bool is_usm_;

        sycl::buffer<one_vm::status, 1> buf_status_;
        one_vm::status * usm_status_;
        int64_t len_;

        one_vm::status status_to_fix_;
        T fixup_value_;
        bool copy_sign_;

        error_handler():
            enabled_ { false },
            is_usm_  { false },

            buf_status_ { sycl::buffer<one_vm::status, 1> { 1 } },
            usm_status_ { nullptr },
            len_ { 0 },
            status_to_fix_ { one_vm::status::not_defined },
            fixup_value_ { T {} },
            copy_sign_ { false }
            { }

        error_handler(one_vm::status status_to_fix, T fixup_value, bool copy_sign = false):
            enabled_ { true },
            is_usm_  { false },

            buf_status_ { sycl::buffer<one_vm::status, 1> { 1 } },
            usm_status_ { nullptr },
            len_ { 0 },
            status_to_fix_ { status_to_fix },
            fixup_value_ { fixup_value },
            copy_sign_ { copy_sign }
            { }

        error_handler(one_vm::status * array, std::int64_t len = 1, one_vm::status status_to_fix = one_vm::status::not_defined, T fixup_value = {}, bool copy_sign = false):
            enabled_ { true },
            is_usm_  { true },

            buf_status_ { sycl::buffer<one_vm::status, 1> { 1 } },
            usm_status_ { array },
            len_ { len },
            status_to_fix_ { status_to_fix },
            fixup_value_ { fixup_value },
            copy_sign_ { copy_sign }
            { }

        error_handler(sycl::buffer<one_vm::status, 1> & buf, std::int64_t len = 1, one_vm::status status_to_fix = one_vm::status::not_defined, T fixup_value = {}, bool copy_sign = false):
            enabled_ { true },
            is_usm_  { false },

            buf_status_ { buf },
            usm_status_ { nullptr },
            len_ { len },
            status_to_fix_ { status_to_fix },
            fixup_value_ { fixup_value },
            copy_sign_ { copy_sign }
            { }

    }; // struct error_handler
} // namespace detail

    using detail::error_handler;

    // Service functions
    __MKL_VM_EXPORT oneapi::mkl::vm::mode get_mode(cl::sycl::queue & queue);
    __MKL_VM_EXPORT oneapi::mkl::vm::mode set_mode(cl::sycl::queue & queue, oneapi::mkl::vm::mode new_mode);

    __MKL_VM_EXPORT oneapi::mkl::vm::status  get_status(cl::sycl::queue & queue);
    __MKL_VM_EXPORT oneapi::mkl::vm::status  set_status(cl::sycl::queue & queue, oneapi::mkl::vm::status new_status);
    __MKL_VM_EXPORT oneapi::mkl::vm::status  clear_status(cl::sycl::queue & queue);


    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event abs(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event acos(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event acosh(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event acospi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event acospi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event acospi(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event acospi(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event arg(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event arg(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event arg(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event arg(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event asin(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event asinh(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event asinpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event asinpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event asinpi(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event asinpi(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event atan2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event atan2(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan2(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event atan2pi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan2pi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event atan2pi(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atan2pi(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event atanh(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event atanpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atanpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event atanpi(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event atanpi(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event cbrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event cbrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event cbrt(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event cbrt(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event cdfnorm(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cdfnorm(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event cdfnorm(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cdfnorm(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event cdfnorminv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cdfnorminv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event cdfnorminv(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cdfnorminv(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event ceil(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event ceil(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event ceil(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event ceil(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event cis(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cis(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});

    __MKL_VM_EXPORT cl::sycl::event cis(cl::sycl::queue & q, std::int64_t n, float * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cis(cl::sycl::queue & q, std::int64_t n, double * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});


    __MKL_VM_EXPORT cl::sycl::event conj(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event conj(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event conj(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event conj(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event copysign(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event copysign(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event copysign(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event copysign(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cos(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event cosd(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosd(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event cosd(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosd(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cosh(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event cospi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cospi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event cospi(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event cospi(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event div(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event erf(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event erf(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event erf(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event erf(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event erfc(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event erfc(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event erfc(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event erfc(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event erfcinv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event erfcinv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event erfcinv(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event erfcinv(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event erfinv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event erfinv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event erfinv(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event erfinv(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event exp2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event exp2(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp2(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event exp10(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp10(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event exp10(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event exp10(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event expint1(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event expint1(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event expint1(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event expint1(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event expm1(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event expm1(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event expm1(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event expm1(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event fdim(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event fdim(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event fdim(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event fdim(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event floor(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event floor(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event floor(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event floor(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event fmax(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event fmax(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event fmax(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event fmax(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event fmin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event fmin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event fmin(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event fmin(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event fmod(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event fmod(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event fmod(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event fmod(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event frac(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event frac(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event frac(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event frac(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event hypot(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event hypot(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event hypot(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event hypot(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event inv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event inv(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event inv(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event inv(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event invcbrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event invcbrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event invcbrt(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event invcbrt(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event invsqrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event invsqrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event invsqrt(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event invsqrt(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event lgamma(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event lgamma(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event lgamma(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event lgamma(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event linearfrac(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, float c, float d, float e, float f, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event linearfrac(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, double c, double d, double e, double f, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event linearfrac(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float c, float d, float e, float f, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event linearfrac(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double c, double d, double e, double f, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event ln(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event log2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event log2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event log2(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event log2(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event logb(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event logb(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event logb(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event logb(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event log10(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event log1p(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event log1p(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event log1p(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event log1p(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event maxmag(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event maxmag(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event maxmag(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event maxmag(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event minmag(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event minmag(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event minmag(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event minmag(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event modf(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, cl::sycl::buffer<float> & z, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event modf(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, cl::sycl::buffer<double> & z, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event modf(cl::sycl::queue & q, std::int64_t n, float * a, float * y, float * z, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event modf(cl::sycl::queue & q, std::int64_t n, double * a, double * y, double * z, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event mul(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event mulbyconj(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event mulbyconj(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});

    __MKL_VM_EXPORT cl::sycl::event mulbyconj(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event mulbyconj(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});


    __MKL_VM_EXPORT cl::sycl::event nearbyint(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event nearbyint(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event nearbyint(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event nearbyint(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event nextafter(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event nextafter(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event nextafter(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event nextafter(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event pow2o3(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event pow2o3(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event pow2o3(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event pow2o3(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event pow3o2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow3o2(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event pow3o2(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event pow3o2(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event powr(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event powr(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event powr(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event powr(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, std::complex<float> b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, std::complex<double> b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, float b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, double b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, float * a, float b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event powx(cl::sycl::queue & q, std::int64_t n, double * a, double b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event remainder(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event remainder(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event remainder(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event remainder(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event rint(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event rint(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event rint(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event rint(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event round(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event round(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event round(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event round(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sin(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event sincos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, cl::sycl::buffer<float> & z, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sincos(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, cl::sycl::buffer<double> & z, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sincos(cl::sycl::queue & q, std::int64_t n, float * a, float * y, float * z, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sincos(cl::sycl::queue & q, std::int64_t n, double * a, double * y, double * z, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event sind(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sind(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sind(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sind(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinh(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event sinpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sinpi(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sinpi(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event sqr(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event sqr(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event sqr(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event sqr(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sqrt(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event sub(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {});
    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tan(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event tand(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tand(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event tand(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tand(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event tanh(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);


    __MKL_VM_EXPORT cl::sycl::event tanpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tanpi(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event tanpi(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tanpi(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event tgamma(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tgamma(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});

    __MKL_VM_EXPORT cl::sycl::event tgamma(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {});
    __MKL_VM_EXPORT cl::sycl::event tgamma(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {});


    __MKL_VM_EXPORT cl::sycl::event trunc(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event trunc(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);

    __MKL_VM_EXPORT cl::sycl::event trunc(cl::sycl::queue & q, std::int64_t n, float * a, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);
    __MKL_VM_EXPORT cl::sycl::event trunc(cl::sycl::queue & q, std::int64_t n, double * a, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined);



} // namespace vm
} // namespace mkl
} // namespace oneapi
#endif // #ifndef _ONEAPI_MKL_VM_HPP
