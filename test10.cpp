// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <CopCore/Ranluxpp.h>
#include <AdePT/SparseVector.h>

#include <VecGeom/base/Stopwatch.h>

/** The test fills a sparse vector with tracks having random energy. It demonstrates allocation,
concurrent distribution of elements, selection based on a lambda predicate function, gathering
of used slots in a selection vector, compacting elements by copy-constructing in a second sparse vector.
 */

/// A simple track
struct Track_t {
  using Rng_t = RanluxppDouble;

  Rng_t rng;
  float energy{0.};
  bool alive{true};

  // a default constructor is not necessarily needed
  // constructor parameters (or copy constructor) can be passed via SparseVectorImplementation::next_free()
  Track_t(unsigned itr)
  {
    rng.SetSeed(itr);
    energy = (float)rng.Rndm();
  }
};

// some utility kernels for filling the vector concurrently and printing info (vector resides on device)

void fill_tracks(adept::SparseVectorInterface<Track_t> *vect1_ptr, int num_elem, sycl::nd_item<3> item_ct1)
{
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
  if (tid >= num_elem) return;
  // parameters of next_free are passed to the matching constructor called in place
  Track_t *track = vect1_ptr->next_free(tid);
  if (!track) COPCORE_EXCEPTION("Out of vector space");
}

void print_tracks(adept::SparseVectorInterface<Track_t> *tracks, int start, int num, const sycl::stream &stream_ct1)
{
  const int nshared = tracks->size();
  stream_ct1 << " data: ";
  for (int i = start; i < start + num && i < nshared; ++i) {
    /*
    DPCT1015:0: Output needs adjustment.
    */
    stream_ct1 << " %.2f";
    if (!tracks->is_used(i)) stream_ct1 << "x";
  }
  stream_ct1 << "...\n";
}

void print_selected_tracks(adept::SparseVectorInterface<Track_t> *tracks, const unsigned *selection,
                                      const unsigned *n_selected, int start, int num, const sycl::stream &stream_ct1)
{
  /*
  DPCT1015:1: Output needs adjustment.
  */
  stream_ct1 << "selected %d tracks:\n > ";
  int limit = sycl::min(*n_selected, (unsigned int)(start + num));
  for (int i = start; i < limit; ++i) {
    /*
    DPCT1015:2: Output needs adjustment.
    */
    stream_ct1 << "%.2f ";
  }
  stream_ct1 << "...\n";
}

void reset_selection(unsigned *nselected)
{
  *nselected = 0;
}

template <typename Vector_t>
void print_vector(int iarr, Vector_t *vect, const sycl::stream &stream_ct1)
{
  /*
  DPCT1015:3: Output needs adjustment.
  */
  stream_ct1 << "=== vect %d: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n";
}

template <typename Vector_t, typename Function>
void get_vector_data(const Vector_t *vect, Function vect_func, int *data)
{
  // data should be allocated in managed memory, vect_func should call a getter of Vector_t
  *data = vect_func(vect);
}

/// Test performance-critical SparseVector operations, executing as kernels. The syncronization
/// operations exposed are only for timing purposes, the operations are valid also without.
//____________________________________________________________________________________________________
int main(void)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1        = dev_ct1.default_queue();
  constexpr int VectorSize = 1 << 20;
  int ntracks              = 1000000;
  using Vector_t = adept::SparseVector<Track_t, VectorSize>; // 1<<16 is the default vector size if parameter omitted
  using VectorInterface = adept::SparseVectorInterface<Track_t>;

  vecgeom::Stopwatch timer;

  Vector_t *vect1_ptr_d, *vect2_ptr_d;
  unsigned *sel_vector_d;
  unsigned *nselected_hd;
  printf("Running on %d tracks. Size of adept::SparseVector<Track_t, %d> = %lu\n", ntracks, VectorSize,
         sizeof(Vector_t));
  // allocation can be done on device or managed memory
  /*
  DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((vect1_ptr_d = sycl::malloc_device<Vector_t>(1, q_ct1), 0));
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((vect2_ptr_d = sycl::malloc_device<Vector_t>(1, q_ct1), 0));
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sel_vector_d = sycl::malloc_device<unsigned int>(VectorSize, q_ct1), 0));
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((nselected_hd = sycl::malloc_shared<unsigned int>(1, q_ct1), 0));

  // managed variables to read state from device
  int *nshared, *nused, *nselected;
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((nshared = sycl::malloc_shared<int>(2, q_ct1), 0));
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((nused = sycl::malloc_shared<int>(2, q_ct1), 0));
  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((nselected = sycl::malloc_shared<int>(2, q_ct1), 0));

  // static allocator for convenience
  Vector_t::MakeInstanceAt(vect1_ptr_d);
  Vector_t::MakeInstanceAt(vect2_ptr_d);

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       reset_selection(nselected_hd);
                     });
  });

  // Construct and distribute tracks concurrently
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  timer.Start();
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (ntracks + 127) / 128) * sycl::range<3>(1, 1, 128),
                                       sycl::range<3>(1, 1, 128)),
                     [=](sycl::nd_item<3> item_ct1) {
                       fill_tracks(vect1_ptr_d, ntracks, item_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_vector_data(
                           vect1_ptr_d,
                           [] __device__(const VectorInterface *arr) {
                             return arr->size();
                           },
                           nshared);
                     });
  });
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  auto time_fill = timer.Stop();
  std::cout << "time_construct_and_share = " << time_fill << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(1, vect1_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_tracks(vect1_ptr_d, 0, 32, stream_ct1);
                     });
  }); // print just first 32 tracks
  int nfilled = *nshared;
  if (nfilled != ntracks) {
    std::cerr << "Error in next_free.\n";
    return 1;
  }

  // Select tracks with energy < 0.2
  // *** note that we can use any device predicate function with the prototype:
  //   __device__ bool func(int, const Vector_t*) // index in the vector and const vector pointer
  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  timer.Start();
  auto select_func = [] (int i, const VectorInterface *arr) { return ((*arr)[i].energy < 0.2); };
  VectorInterface::select(vect1_ptr_d, select_func, sel_vector_d, nselected_hd);
  /*
  DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  auto time_select = timer.Stop();
  int nselected1   = *nselected_hd;
  std::cout << "\ntime_select for " << nselected1 << " tracks with (energy < 0.2) = " << time_select << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(1, vect1_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_selected_tracks(vect1_ptr_d, sel_vector_d, nselected_hd, 0, 32, stream_ct1);
                     });
  });
  if (nselected1 == 0) {
    std::cerr << "Error in select: 0 tracks.\n";
    return 2;
  }

  // Release the tracks we just selected, creating holes in the vector
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  timer.Start();
  VectorInterface::release_selected(vect1_ptr_d, sel_vector_d, nselected_hd);
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_vector_data(
                           vect1_ptr_d,
                           [] __device__(const VectorInterface *arr) {
                             return arr->size_used();
                           },
                           nused);
                     });
  });
  /*
  DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  auto time_release = timer.Stop();
  std::cout << "\ntime_release_selected = " << time_release << "   nused = " << *nused << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(1, vect1_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_tracks(vect1_ptr_d, 0, 32, stream_ct1);
                     });
  });
  int nused_after_release = *nused;
  if ((nselected1 + nused_after_release) != ntracks) {
    std::cerr << "Error in release_selected.\n";
    return 3;
  }

  // Demonstrate select_and_move functionality
  /*
  DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  timer.Start();
  // a fuction selecting tracks having energy > 0.8. We move these tracks in a second vector
  auto select2_func = [] (int i, const VectorInterface *arr) { return ((*arr)[i].energy > 0.8); };
  //===
  VectorInterface::select_and_move(vect1_ptr_d, select2_func, vect2_ptr_d, nselected_hd);
  //===
  auto time_select_and_move = timer.Stop();
  /*
  DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  q_ct1.submit([&](sycl::handler &cgh) {
    auto nused_ct2 = &nused[0];

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_vector_data(
                           vect1_ptr_d,
                           [] __device__(const VectorInterface *arr) {
                             return arr->size_used();
                           },
                           nused_ct2);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    auto nused_ct2 = &nused[1];

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_vector_data(
                           vect2_ptr_d,
                           [] __device__(const VectorInterface *arr) {
                             return arr->size_used();
                           },
                           nused_ct2);
                     });
  });
  /*
  DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  std::cout << "\ntime_select_and_move (energy > 0.8) = " << time_select_and_move << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(1, vect1_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_tracks(vect1_ptr_d, 0, 32, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(2, vect2_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_tracks(vect2_ptr_d, 0, 32, stream_ct1);
                     });
  });
  // Check the moved tracks
  int nused_after_move  = nused[0];
  int nused_after_move2 = nused[1];
  if ((nused_after_release - nused_after_move) != nused_after_move2) {
    std::cerr << "Error in select_and_move.\n";
    return 4;
  }

  // Demonstrate a common selection method that should be used when the vector is fragmented.
  /*
  DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  timer.Start();
  VectorInterface::select_used(vect1_ptr_d, sel_vector_d, nselected_hd);
  /*
  DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  auto time_select_used = timer.Stop();
  std::cout << "\ntime_select_used = " << time_select_used << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_selected_tracks(vect1_ptr_d, sel_vector_d, nselected_hd, 0, 32, stream_ct1);
                     });
  });
  if (*nselected_hd != nused_after_move) {
    std::cerr << "Error in select_used.\n";
    return 5;
  }

  // Compact used elements by copying them into a destination vector. The stage above should be preferred
  // if the sparsity is small, while this one is preffered for high sparsity. See SparseVector header
  // for the definition of sparsity, shared and selected fractions.
  /*
  DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  timer.Start();
  VectorInterface::compact(vect1_ptr_d, vect2_ptr_d, nselected_hd);
  /*
  DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));
  auto time_compact = timer.Stop();
  q_ct1.submit([&](sycl::handler &cgh) {
    auto nused_ct2 = &nused[0];

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_vector_data(
                           vect1_ptr_d,
                           [] __device__(const VectorInterface *arr) {
                             return arr->size_used();
                           },
                           nused_ct2);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    auto nused_ct2 = &nused[1];

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_vector_data(
                           vect2_ptr_d,
                           [] __device__(const VectorInterface *arr) {
                             return arr->size_used();
                           },
                           nused_ct2);
                     });
  });
  std::cout << "\ntime_compact = " << time_compact << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(1, vect1_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_vector(2, vect2_ptr_d, stream_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       print_tracks(vect2_ptr_d, 0, 32, stream_ct1);
                     });
  });
  if ((nused[0] != 0) || (nused[1] != nused_after_move2 + nused_after_move)) {
    std::cerr << "Error in compact.\n";
    return 6;
  }

  /*
  DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(vect1_ptr_d, q_ct1), 0));
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(vect2_ptr_d, q_ct1), 0));
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(sel_vector_d, q_ct1), 0));

  return 0;
}
