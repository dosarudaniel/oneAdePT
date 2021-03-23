// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file CopCore/Global.h
 * @brief CopCore global macros and types
 */

#ifndef COPCORE_1GLOBAL_H_
#define COPCORE_1GLOBAL_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <type_traits>
#include <CopCore/1/Macros.h>

#define COPCORE_IMPL cpp

namespace copcore {

/** @brief Backend types enumeration */
enum BackendType { CPU = 0, CUDA, HIP };

/** @brief CUDA error checking */

inline void error_check(int, const char *, int) {}

#define COPCORE_CUDA_CHECK(err)

/** @brief Trigger a runtime error depending on the backend */
#define COPCORE_EXCEPTION(message) throw std::runtime_error(message)

/** @brief Check if pointer id device-resident */
inline bool is_device_pointer(void *ptr) { return false; }

/** @brief Get number of SMs on the current device */

inline int get_num_SMs() { return 0; }

template <BackendType T>
struct StreamType {
  using value_type = int;
  static void CreateStream(value_type &stream) { stream = 0; }
};

/** @brief Getting the backend name in templated constructs */
template <typename Backend>
const char *BackendName(Backend const &backend)
{
  switch (backend) {
  case BackendType::CPU:
    return "BackendType::CPU";
  case BackendType::CUDA:
    return "BackendType::CUDA";
  case BackendType::HIP:
    return "BackendType::HIP";
  default:
    return "Unknown backend";
  };
};

} // End namespace copcore

/** @brief Macro to template-specialize on a specific compile-time requirement */
#define COPCORE_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type * = nullptr

/** @brief macro to declare device callable functions usable in executors */
#define COPCORE_CALLABLE_FUNC(FUNC) auto _ptr_##FUNC = FUNC;

/** @brief macro to pass callable function to executors */
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC) auto HVAR = FUNC;

#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC) auto HVAR = NAMESPACE::FUNC;

#endif // COPCORE_GLOBAL_H_
