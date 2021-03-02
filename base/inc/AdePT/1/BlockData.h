// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file BlockData.h
 * @brief Templated data structure storing a contiguous block of data.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_1BLOCKDATA_H_
#define ADEPT_1BLOCKDATA_H_

#include <CopCore/1/CopCore.h>
#include <AdePT/1/Atomic.h>
#include <AdePT/1/mpmc_bounded_queue.h>

namespace adept {

/** @brief A container of data that adopts memory. It is caller responsibility to allocate
  at least SizeOfInstance bytes for a single object, and SizeOfAlignAware for multiple objects.
  Write access to data elements in the block is given atomically, up to the capacity.
 */
template <typename Type>
class BlockData : protected copcore::VariableSizeObjectInterface<BlockData<Type>, Type> {

public:
  using AtomicInt_t = adept::Atomic_t<int>;
  using Queue_t     = adept::mpmc_bounded_queue<int>;
  using Value_t     = Type;
  using Base_t      = copcore::VariableSizeObjectInterface<BlockData<Value_t>, Value_t>;
  using ArrayData_t = copcore::VariableSizeObj<Value_t>;

private:
  int fCapacity{0};         ///< Maximum number of elements
  AtomicInt_t fNbooked;     ///< Number of booked elements
  AtomicInt_t fNused;       ///< Number of used elements
  Queue_t *fHoles{nullptr}; ///< Queue of holes
  ArrayData_t fData;        ///< Data follows, has to be last

private:
  friend Base_t;

  /** @brief Functions required by VariableSizeObjectInterface */
  ArrayData_t &GetVariableData() { return fData; }
  const ArrayData_t &GetVariableData() const { return fData; }

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function
  BlockData(size_t nvalues) : fCapacity(nvalues), fData(nvalues)
  {
    char *address = (char *)this + Base_t::SizeOfAlignAware(nvalues) - BlockData<Type>::SizeOfExtra(nvalues);
    fHoles        = (Queue_t *)address;
    Queue_t::MakeInstanceAt(nvalues, address);
  }

  BlockData(BlockData const &other) : BlockData(other.fCapacity, other) {}

  BlockData(size_t new_size, BlockData const &other) : Base_t(other), fCapacity(new_size), fData(new_size, other.fData)
  {
    char *address = (char *)this + Base_t::SizeOfAlignAware(new_size) - BlockData<Type>::SizeOfExtra(new_size);
    fHoles        = (Queue_t *)address;
    Queue_t::MakeCopyAt(new_size, *other.fHoles, address);
  }

  ~BlockData() {}

  /** @brief Returns the size in bytes of extra queue data needed by BlockData object with given capacity */
  static size_t SizeOfExtra(int capacity) { return Queue_t::SizeOfAlignAware(capacity); }

public:
  ///< Enumerate the part of the private interface, we want to expose.
  using Base_t::MakeCopy;
  using Base_t::MakeCopyAt;
  using Base_t::MakeInstance;
  using Base_t::MakeInstanceAt;
  using Base_t::ReleaseInstance;
  using Base_t::SizeOf;
  using Base_t::SizeOfAlignAware;

  /** @brief Returns the size in bytes of a BlockData object with given capacity */
  static size_t SizeOfInstance(int capacity) { return Base_t::SizeOf(capacity); }

  /** @brief Maximum number of elements */
  int Capacity() const { return fCapacity; }

  /** @brief Clear the content */
  void Clear()
  {
    fNused.store(0);
    fNbooked.store(0);
  }

  /** @brief Read-only index operator */
  Type const &operator[](const int index) const { return fData[index]; }

  /** @brief Read/write index operator */
  Type &operator[](const int index) { return fData[index]; }

  /** @brief Dispatch next free element, nullptr if none left */
  Type *NextElement()
  {
    // Try to get a hole index if any
    int index = -1;
    if (fHoles->dequeue(index)) {
      fNused++;
      return &fData[index];
    }
    index = fNbooked.fetch_add(1);
    if (index >= fCapacity) return nullptr;
    fNused++;
    return &fData[index];
  }

  /** @brief Release an element */
  void ReleaseElement(int index)
  {
    // No checks currently done that the index was given via NextElement
    // or that it wasn't already released. Either case will produce errors.
    fNused--;
    fHoles->enqueue(index);
  }

  /** @brief Number of elements currently distributed */
  int GetNused() { return fNused.load(); }

  /** @brief Number of holes in the block */
  int GetNholes() { return fHoles->size(); }

  /** @brief Check if container is fully distributed */
  bool IsFull() const { return (GetNused() == fCapacity); }

}; // End BlockData
} // End namespace adept

#endif // ADEPT_1BLOCKDATA_H_
