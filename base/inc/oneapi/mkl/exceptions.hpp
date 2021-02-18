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

#ifndef _MKL_EXCEPTIONS_HPP__
#define _MKL_EXCEPTIONS_HPP__

#include <exception>
#include <string>
#include <CL/sycl.hpp>

namespace oneapi {
namespace mkl {
    class exception : public std::exception {
        std::string msg_;
        public:
            exception(const std::string &domain, const std::string &function, const std::string &info = "") : std::exception() {
                msg_ = std::string("oneMKL: ") + domain +
                       ((domain.length() != 0 && function.length() != 0) ? "/" : "") + function +
                       ((info.length() != 0)
                        ? (((domain.length() + function.length() != 0) ? ": " : "") + info)
                        : "");
            }

            const char* what() const noexcept {
                return msg_.c_str();
            }
    };

    class unsupported_device : public oneapi::mkl::exception {
        public:
            unsupported_device(const std::string &domain, const std::string &function, const cl::sycl::device &device)
                : oneapi::mkl::exception(domain, function, device.get_info<cl::sycl::info::device::name>()+" is not supported") {
            }
    };

    class host_bad_alloc : public oneapi::mkl::exception {
        public:
            host_bad_alloc(const std::string &domain, const std::string &function)
                : oneapi::mkl::exception(domain, function, "cannot allocate memory on host") {
            }
    };

    class device_bad_alloc : public oneapi::mkl::exception {
        public:
            device_bad_alloc(const std::string &domain, const std::string &function, const cl::sycl::device &device)
                : oneapi::mkl::exception(domain, function, "cannot allocate memory on "+device.get_info<cl::sycl::info::device::name>()) {
            }
    };

    class unimplemented : public oneapi::mkl::exception {
        public:
            unimplemented(const std::string &domain, const std::string &function, const std::string &info = "")
                : oneapi::mkl::exception(domain, function, "function is not implemented "+info) {
            }
    };

    class invalid_argument : public oneapi::mkl::exception {
        public:
            invalid_argument(const std::string &domain, const std::string &function, const std::string &info = "")
                : oneapi::mkl::exception(domain, function, "invalid argument "+info) {
            }
    };

    class uninitialized : public oneapi::mkl::exception {
        public:
            uninitialized(const std::string &domain, const std::string &function, const std::string &info = "")
                : oneapi::mkl::exception(domain, function, "handle/descriptor is not initialized "+info) {
            }
    };

    class computation_error : public oneapi::mkl::exception {
        public:
            computation_error(const std::string &domain, const std::string &function, const std::string &info = "")
                : oneapi::mkl::exception(domain, function, "computation error"+((info.length() != 0) ? (": "+info) : "")) {
            }
    };

    class batch_error : public oneapi::mkl::exception {
        public:
            batch_error(const std::string &domain, const std::string &function, const std::string &info = "")
                : oneapi::mkl::exception(domain, function, "batch error"+((info.length() != 0) ? (": "+info) : "")) {
            }
    };

} // namespace mkl
} // namespace oneapi

#endif // _MKL_EXCEPTIONS_HPP__

