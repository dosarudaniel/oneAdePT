// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_queue.cu
 * @brief Unit test for queue operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cassert>
#include <AdePT/mpmc_bounded_queue.h>

// Kernel function to perform atomic addition
void pushData(adept::mpmc_bounded_queue<int> *queue, sycl::nd_item<3> item_ct1)
{
  // Push the thread index in the queue
  int id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
  queue->enqueue(id);
}

// Kernel function to dequeue a value and add it atomically
void popAndAdd(adept::mpmc_bounded_queue<int> *queue, adept::Atomic_t<unsigned long long> *sum)
{
  // Push the thread index in the queue
  int id = 0;
  if (!queue->dequeue(id)) id = 0;
  sum->fetch_add(id);
}

///______________________________________________________________________________________
int main(void)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1        = dev_ct1.default_queue();
  using Queue_t      = adept::mpmc_bounded_queue<int>;
  using AtomicLong_t = adept::Atomic_t<unsigned long long>;

  const char *result[2] = {"FAILED", "OK"};
  bool success          = true;
  // Define the kernels granularity: 10K blocks of 32 treads each
  sycl::range<3> nblocks(1, 1, 1000), nthreads(1, 1, 32);

  int capacity      = 1 << 15; // 32768 - accomodates values pushed by all threads
  size_t buffersize = Queue_t::SizeOfInstance(capacity);
  char *buffer      = nullptr;
  buffer            = (char *)sycl::malloc_shared(buffersize, q_ct1);
  auto queue = Queue_t::MakeInstanceAt(capacity, buffer);

  char *buffer_atomic = nullptr;
  buffer_atomic       = (char *)sycl::malloc_shared(sizeof(AtomicLong_t), q_ct1);
  auto sum = new (buffer_atomic) AtomicLong_t;

  bool testOK = true;
  std::cout << "   test_queue ... ";
  // Allow memory to reach the device
  dev_ct1.queues_wait_and_throw();
  // Launch a kernel queueing thread id's
  /*
  DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
      pushData(queue, item_ct1);
    });
  });
  // Allow all warps in the stream to finish
  dev_ct1.queues_wait_and_throw();
  // Make sure all threads managed to queue their id
  testOK &= queue->size() == nblocks[2] * nthreads[2];
  // Launch a kernel top collect queued data
  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
      popAndAdd(queue, sum);
    });
  });
  // Wait work to finish and memory to reach the host
  dev_ct1.queues_wait_and_throw();
  // Check if all data was dequeued
  testOK &= queue->size() == 0;
  // Check if the sum of all dequeued id's matches the sum of thread indices
  unsigned long long sumref = 0;
  for (auto i = 0; i < nblocks[2] * nthreads[2]; ++i)
    sumref += i;
  testOK &= sum->load() == sumref;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  sycl::free(buffer, q_ct1);
  sycl::free(buffer_atomic, q_ct1);
  if (!success) return 1;
  return 0;
}
