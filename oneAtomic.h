// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Utils.h
 * @brief Portable atomic structure.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_ATOMIC_H_
#define ADEPT_ATOMIC_H_

#include <CopCore/Global.h>
#include <cassert>

namespace adept {
/**
 * @brief A portable atomic type. Not all base types are supported by CUDA.
 */
  template <typename Type>
  struct AtomicBase_t {

  using AtomicType_t = sycl::ONEAPI::atomic_ref<Type,
		       sycl::ONEAPI::memory_order::seq_cst,
		       sycl::ONEAPI::memory_scope::work_item,
		       sycl::access::address_space::global_space>;						

    AtomicType_t fData{0}; ///< Atomic data

  /** @brief Constructor taking an address */
    AtomicBase_t() : fData{0} {}

   /** @brief Copy constructor */
   AtomicBase_t(AtomicBase_t const &other) { store(other.load()); }

  /** @brief Assignment */
  AtomicBase_t &operator=(AtomicBase_t const &other) { store(other.load()); }

  /** @brief Emplace the data at a given address */
  static AtomicBase_t *MakeInstanceAt(void *addr)
  {
    assert(addr != nullptr && "cannot allocate at nullptr address");
    assert((((unsigned long long)addr) % alignof(AtomicType_t)) == 0 && "addr does not satisfy alignment");
    AtomicBase_t *obj = new (addr) AtomicBase_t();
    return obj;
  }

  /** @brief Atomically replaces the current value with desired. */
  void store(Type desired) { fData.store(desired); }

  /** @brief Atomically loads and returns the current value of the atomic variable. */
  Type load() const { return fData.load(); }

  /** @brief Atomically replaces the underlying value with desired. */
  Type exchange(Type desired) { return fData.exchange(desired); }

  /** @brief Atomically compares the stored value with the expected one, and if equal stores desired.
   * If not loads the old value into expected. Returns true if swap was successful. */
  bool compare_exchange_strong(Type &expected, Type desired)
  {
    return fData.compare_exchange_strong(expected, desired);
  }

};

/** @brief Atomic_t generic implementation specialized using SFINAE mechanism */
template <typename Type, typename Enable = void>
struct Atomic_t;

/** @brief Specialization for integral types. */
template <typename Type>
struct Atomic_t<Type, typename std::enable_if<std::is_integral<Type>::value>::type> : public AtomicBase_t<Type> {
  using AtomicBase_t<Type>::fData;

  /// Standard library atomic operations for integral types

  /** @brief Atomically assigns the desired value to the atomic variable. */
  Type operator=(Type desired) { return fData.operator=(desired); }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  Type fetch_add(Type arg) { return fData.fetch_add(arg); }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  Type fetch_sub(Type arg) { return fData.fetch_sub(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise AND of the value and arg. */
  Type fetch_and(Type arg) { return fData.fetch_and(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise OR of the value and arg. */
  Type fetch_or(Type arg) { return fData.fetch_or(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise XOR of the value and arg. */
  Type fetch_xor(Type arg) { return fData.fetch_xor(arg); }

  /** @brief Atomically replaces the current value with the result of MAX of the value and arg. */
  Type fetch_max(Type arg) { return fData.fetch_max(&fData, arg); }

  /** @brief Atomically replaces the current value with the result of MIN of the value and arg. */
  Type fetch_min(Type arg) { return fData.fetch_min(&fData, arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator+=(Type arg) { return (fetch_add(arg) + arg); }

  /** @brief Performs atomic subtract. */
  Type operator-=(Type arg) { return (fetch_sub(arg) - arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  Type operator--(int) { return fetch_sub(1); }

}; // End specialization for integral types of Atomic_t

/** @brief Specialization for non-integral types. */
template <typename Type>
struct Atomic_t<Type, typename std::enable_if<!std::is_integral<Type>::value>::type> : public AtomicBase_t<Type> {
  using Base_t = AtomicBase_t<Type>;
  using Base_t::fData;

  /** @brief Atomically assigns the desired value to the atomic variable. */
  Type operator=(Type desired) { return fData.operator=(desired); }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  Type fetch_add(Type arg)
  {
    Type current = fData.load();
    while (!fData.compare_exchange_weak(current, current + arg))
      ;
    return current;
  }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  Type fetch_sub(Type arg) { return fetch_add(-arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator+=(Type arg) { return (fetch_add(arg) + arg); }

  /** @brief Performs atomic subtract. */
  Type operator-=(Type arg) { return (fetch_sub(arg) - arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  Type operator--(int) { return fetch_sub(1); }

};

} // End namespace adept
#endif // ADEPT_ATOMIC_H_
