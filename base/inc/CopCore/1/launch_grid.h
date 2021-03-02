// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file launch_grid.h
 * @brief A kernel launch grid of blocks/threads.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_1LAUNCH_GRID_H_
#define ADEPT_1LAUNCH_GRID_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <CopCore/1/Global.h>

namespace copcore {

/** @brief Helper allowing to handle kernel launch configurations as compact type */
template <BackendType backend>
class launch_grid {
};

template <>
class launch_grid<BackendType::CPU> {
private:
  int fGrid[2]; ///< Number of blocks/threads per block

public:
  launch_grid(int n_blocks, int n_threads) : fGrid{n_blocks, n_threads} {}

  /** @brief Access either block [0] or thread [1] grid */
  
  int operator[](int index) { return fGrid[index]; }

  /** @brief Access either block [0] or thread [1] grid */
  
  int operator[](int index) const { return fGrid[index]; }

}; // End class launch_grid<BackendType::CPU>

template <>
class launch_grid<BackendType::CUDA> {
private:
  sycl::range<3> fGrid[2]; ///< Block and thread index grid

public:
  /** @brief Construct from block and thread grids */

  launch_grid(const sycl::range<3> &block_index,
              const sycl::range<3> &thread_index)
      : fGrid{block_index, thread_index} {}

  /** @brief Default constructor */

  //  launch_grid() : launch_grid(sycl::range<3>(), sycl::range<3>()) {}

  /** @brief Access either block [0] or thread [1] grid */

  sycl::range<3> &operator[](int index) { return fGrid[index]; }

  /** @brief Access either block [0] or thread [1] grid */

  const sycl::range<3> &operator[](int index) const { return fGrid[index]; }
}; // End class launch_grid<BackendType::CUDA>

} // End namespace copcore

#endif // ADEPT_LAUNCH_GRID_H_
