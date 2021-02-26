// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_block.cu
 * @brief Unit test for the BlockData concurrent container.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cassert>
#include <AdePT/1/BlockData.h>

struct MyTrack {
  int index{0};
  double pos[3]{0};
  double dir[3]{0};
  bool flag1;
  bool flag2;
};

// Kernel function to process the next free track in a block
void testTrackBlock(adept::BlockData<MyTrack> *block, sycl::nd_item<3> item_ct1)
{
  auto track = block->NextElement();
  if (!track) return;
  int id       = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
  track->index = id;
}

// Kernel function to process the next free track in a block
void releaseTrack(adept::BlockData<MyTrack> *block, sycl::nd_item<3> item_ct1)
{
  int id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
  block->ReleaseElement(id);
}

///______________________________________________________________________________________
int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  using Block_t         = adept::BlockData<MyTrack>;
  const char *result[2] = {"FAILED", "OK"};
  // Track capacity of the block
  constexpr int capacity = 1 << 20;

  // Define the kernels granularity: 10K blocks of 32 treads each
  //constexpr sycl::range<3> nblocks(1, 1, 10000), nthreads(1, 1, 32);
  const sycl::range<3> nblocks(1, 1, 10000), nthreads(1, 1, 32);

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block

  bool testOK  = true;
  bool success = true;

  // Test simple allocation/de-allocation on host
  std::cout << "   host allocation MakeInstance ... ";
  auto h_block = Block_t::MakeInstance(1024);
  testOK &= h_block != nullptr;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Test using the slots on the block (more than existing)
  std::cout << "   host NextElement             ... ";
  testOK           = true;
  size_t checksum1 = 0;
  for (auto i = 0; i < 2048; ++i) {
    auto track = h_block->NextElement();
    if (i >= 1024) testOK &= track == nullptr;
    // Assign current index to the current track
    if (track) {
      track->index = i;
      checksum1 += i;
    }
  }
  testOK = h_block->GetNused() == 1024;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Create another block into adopted memory on host
  testOK          = true;
  char *buff_host = new char[Block_t::SizeOfInstance(2048)];
  std::cout << "   host MakeCopyAt              ... ";
  // Test copying a block into another
  auto h_block2    = Block_t::MakeCopyAt(*h_block, buff_host);
  size_t checksum2 = 0;
  for (auto i = 0; i < 1024; ++i) {
    auto track = h_block2->NextElement();
    assert(track);
    checksum2 += track->index;
  }
  testOK = checksum1 == checksum2;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Release some elements end validate
  testOK = true;
  std::cout << "   host ReleaseElement          ... ";
  for (auto i = 0; i < 10; ++i)
    h_block2->ReleaseElement(i);
  testOK &= h_block2->GetNused() == (1024 - 10);
  testOK &= h_block2->GetNholes() == 10;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Release allocated blocks
  Block_t::ReleaseInstance(h_block);  // mandatory, frees memory for blocks allocated with MakeInstance
  Block_t::ReleaseInstance(h_block2); // will not do anything since block adopted memory
  delete[] buff_host;                 // Only this will actually free the memory

  // Create a large block on the device
  testOK = true;
  std::cout << "   host MakeInstanceAt          ... ";
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer     = nullptr;
  /*
  DPCT1004:0: Could not generate replacement.
  */
  
  //cudaMallocManaged(&buffer, blocksize);
  buffer        = (char *)sycl::malloc_shared(blocksize, q_ct1);
  
  auto block = Block_t::MakeInstanceAt(capacity, buffer);
  testOK &= block != nullptr;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  std::cout << "   device NextElement           ... ";
  testOK = true;
  // Allow memory to reach the device
  q_ct1.wait_and_throw();
  // Launch a kernel processing tracks
  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> block_buf_ct0 = dpct::get_buffer_and_offset(block);
    size_t block_offset_ct0                         = block_buf_ct0.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto block_acc_ct0 = block_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
        adept::BlockData<MyTrack> *block_ct0 = (adept::BlockData<MyTrack> *)(&block_acc_ct0[0] + block_offset_ct0);
        testTrackBlock(block_ct0, item_ct1);
      });
    });
  } ///< note that we are passing a host block type allocated on device
    ///< memory - works because the layout is the same
  // Allow all warps to finish
  q_ct1.wait_and_throw();
  // The number of used tracks should be equal to the number of spawned threads
  testOK &= block->GetNused() == nblocks[2] * nthreads[2];
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Compute the sum of assigned track indices, which has to match the sum from 0 to nthreads-1
  // (the execution order is arbitrary, but all thread indices must be distributed)
  unsigned long long counter1 = 0, counter2 = 0;
  testOK = true;
  std::cout << "   device concurrency checksum  ... ";
  for (auto i = 0; i < nblocks[2] * nthreads[2]; ++i) {
    counter1 += i;
    counter2 += (*block)[i].index;
  }
  testOK &= counter1 == counter2;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Now release 32K tracks
  testOK = true;
  std::cout << "   device ReleaseElement        ... ";
  {
    std::pair<dpct::buffer_t, size_t> block_buf_ct0 = dpct::get_buffer_and_offset(block);
    size_t block_offset_ct0                         = block_buf_ct0.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto block_acc_ct0 = block_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1000) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
          [=](sycl::nd_item<3> item_ct1) {
            adept::BlockData<MyTrack> *block_ct0 = (adept::BlockData<MyTrack> *)(&block_acc_ct0[0] + block_offset_ct0);
            releaseTrack(block_ct0, item_ct1);
          });
    });
  }
  q_ct1.wait_and_throw();
  testOK &= block->GetNused() == nblocks[2] * nthreads[2] - 32000;
  testOK &= block->GetNholes() == 32000;
  // Now allocate in the holes
  {
    std::pair<dpct::buffer_t, size_t> block_buf_ct0 = dpct::get_buffer_and_offset(block);
    size_t block_offset_ct0                         = block_buf_ct0.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto block_acc_ct0 = block_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 10) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                       [=](sycl::nd_item<3> item_ct1) {
                         adept::BlockData<MyTrack> *block_ct0 =
                             (adept::BlockData<MyTrack> *)(&block_acc_ct0[0] + block_offset_ct0);
                         testTrackBlock(block_ct0, item_ct1);
                       });
    });
  }
  q_ct1.wait_and_throw();
  testOK &= block->GetNholes() == (32000 - 320);
  std::cout << result[testOK] << "\n";
  sycl::free(buffer, q_ct1);
  if (!success) return 1;
  return 0;
}
