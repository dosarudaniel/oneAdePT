// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <iostream>

#include <AdePT/1/BlockData.h>

using Queue_t = adept::mpmc_bounded_queue<int>;

struct MyTrack {
  int index{0};
  int pdg{0};
  double energy{10};
  double pos[3]{0};
  double dir[3]{1};
  bool flag1;
  bool flag2;
};

struct Scoring {
  adept::Atomic_t<int> secondaries;
  adept::Atomic_t<float> totalEnergyLoss;
  
  Scoring() {}

  static Scoring *MakeInstanceAt(void *addr)
  {
    Scoring *obj = new (addr) Scoring();
    return obj;
  }
};

// kernel function that does transportation
/*
DPCT1032:0: Different generator is used, you may need to adjust the code.
*/
void transport(int n, adept::BlockData<MyTrack> *block, oneapi::mkl::rng::device::philox4x32x10<1> *states,
               Queue_t *queues, sycl::nd_item<3> item_ct1)
{
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    // transport particles
    for (int xyz = 0; xyz < 3; xyz++) {
      (*block)[i].pos[xyz] = (*block)[i].pos[xyz] + (*block)[i].energy * (*block)[i].dir[xyz];
    }
  }
}

// kernel function that assigns next process to the particle
/*
DPCT1032:1: Different generator is used, you may need to adjust the code.
*/
void select_process(adept::BlockData<MyTrack> *block, Scoring *scor, oneapi::mkl::rng::device::philox4x32x10<1> *states,
                    Queue_t *queues[], sycl::nd_item<3> item_ct1)
{
  oneapi::mkl::rng::device::uniform<float> distr_ct1;
  int particle_index = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

  // check if you are not outside the used block
  if (particle_index > block->GetNused() + block->GetNholes()) return;

  // check if the particle is still alive (E>0)
  if ((*block)[particle_index].energy == 0) return;

  // generate random number
  float r = oneapi::mkl::rng::device::generate(distr_ct1, *states);

  if (r > 0.5f) {
    queues[0]->enqueue(particle_index);
  } else {
    queues[1]->enqueue(particle_index);
  }
}

// kernel function that does energy loss
/*
DPCT1032:2: Different generator is used, you may need to adjust the code.
*/
void process_eloss(int n, adept::BlockData<MyTrack> *block, Scoring *scor,
                   oneapi::mkl::rng::device::philox4x32x10<1> *states, Queue_t *queue, sycl::nd_item<3> item_ct1)
{
  int particle_index;
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    if (!queue->dequeue(particle_index)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle_index].energy == 0) return;

    // call the 'process'
    // energy loss
    float eloss = 0.2f * (*block)[particle_index].energy;
    scor->totalEnergyLoss.fetch_add(eloss < 0.001f ? (*block)[particle_index].energy : eloss);
    (*block)[particle_index].energy = (eloss < 0.001f ? 0.0f : ((*block)[particle_index].energy - eloss));

    // if particle dies (E=0) release the slot
    if ((*block)[particle_index].energy < 0.001f) block->ReleaseElement(particle_index);
  }
}

// kernel function that does pair production
/*
DPCT1032:3: Different generator is used, you may need to adjust the code.
*/
void process_pairprod(int n, adept::BlockData<MyTrack> *block, Scoring *scor,
                      oneapi::mkl::rng::device::philox4x32x10<1> *states, Queue_t *queue, sycl::nd_item<3> item_ct1)
{
  int particle_index;
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    if (!queue->dequeue(particle_index)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle_index].energy == 0) return;

    // pair production
    auto secondary_track = block->NextElement();
    assert(secondary_track != nullptr && "No slot available for secondary track");

    float eloss = 0.5f * (*block)[particle_index].energy;
    (*block)[particle_index].energy -= eloss;

    secondary_track->energy = eloss;

    // increase the counter of secondaries
    scor->secondaries.fetch_add(1);
  }
}

/* this GPU kernel function is used to initialize the random states */
/*
DPCT1032:4: Different generator is used, you may need to adjust the code.
*/
void init(oneapi::mkl::rng::device::philox4x32x10<1> *states)
{
  /* we have to initialize the state */
  *states = oneapi::mkl::rng::device::philox4x32x10<1>(0, {0, 0 * 8});
}

//

int main()
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";
   
  /*
  DPCT1032:5: Different generator is used, you may need to adjust the code.
  */
  oneapi::mkl::rng::device::philox4x32x10<1> *state;
  /*
  DPCT1032:6: Different generator is used, you may need to adjust the code.
  */
  state = sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(1, q_ct1);
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       init(state);
                     });
  });
  q_ct1.wait_and_throw();

  // Capacity of the different containers
  constexpr int capacity = 1 << 20;

  using Queue_t = adept::mpmc_bounded_queue<int>;

  constexpr int numberOfProcesses = 3;
  char *buffer[numberOfProcesses];

  Queue_t **queues = nullptr;
  queues           = sycl::malloc_shared<Queue_t *>(numberOfProcesses, q_ct1);

  size_t buffersize = Queue_t::SizeOfInstance(capacity);

  for (int i = 0; i < numberOfProcesses; i++) {
    buffer[i] = nullptr;
    buffer[i] = (char *)sycl::malloc_shared(buffersize, q_ct1);

    queues[i] = Queue_t::MakeInstanceAt(capacity, buffer[i]);
  }

  // Allocate the content of Scoring in a buffer
  char *buffer_scor = nullptr;
  buffer_scor       = (char *)sycl::malloc_shared(sizeof(Scoring), q_ct1);
  Scoring *scor = Scoring::MakeInstanceAt(buffer_scor);
  // Initialize scoring
  // scor->secondaries = 0;
  // scor->totalEnergyLoss = 0;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block

  using Block_t    = adept::BlockData<MyTrack>;
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  buffer2          = (char *)sycl::malloc_shared(blocksize, q_ct1);
  auto block = Block_t::MakeInstanceAt(capacity, buffer2);

  // initializing one track in the block
  auto track    = block->NextElement();
  track->energy = 100.0f;

  // initializing second track in the block
  auto track2    = block->NextElement();
  track2->energy = 30.0f;

  sycl::range<3> nthreads(1, 1, 32);
  sycl::range<3> maxBlocks(1, 1, 10);

  sycl::range<3> numBlocks(1, 1, 1), numBlocks_eloss(1, 1, 1), numBlocks_pairprod(1, 1, 1), numBlocks_transport(1, 1, 1);

  while (block->GetNused()) {

    numBlocks[2] = (block->GetNused() + block->GetNholes() + nthreads[2] - 1) / nthreads[2];

    // here I set the maximum number of blocks

    numBlocks_transport[2] = std::min(numBlocks[2], maxBlocks[2]);


    q_ct1.submit([&](sycl::handler &cgh) {
      auto queues_size_ct0 = queues[2]->size();
      auto queues_ct3      = queues[2];

      cgh.parallel_for(sycl::nd_range<3>(numBlocks_transport * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
        transport(queues_size_ct0, block, state, queues_ct3, item_ct1);
      });
    });

    
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(numBlocks * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
        // select_process(block, scor, state, queues, item_ct1);
      });
    });
    
    q_ct1.wait_and_throw();

    // call the process kernels

    numBlocks_eloss[2]    = std::min((queues[0]->size() + nthreads[2] - 1) / nthreads[2], maxBlocks[2]);
    numBlocks_pairprod[2] = std::min((queues[1]->size() + nthreads[2] - 1) / nthreads[2], maxBlocks[2]);

    q_ct1.submit([&](sycl::handler &cgh) {
      auto queues_size_ct0 = queues[0]->size();
      auto queues_ct4      = queues[0];

      cgh.parallel_for(sycl::nd_range<3>(numBlocks_eloss * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
        // process_eloss(queues_size_ct0, block, scor, state, queues_ct4, item_ct1);
      });
    });

    q_ct1.submit([&](sycl::handler &cgh) {
      auto queues_size_ct0 = queues[1]->size();
      auto queues_ct4      = queues[1];

      cgh.parallel_for(sycl::nd_range<3>(numBlocks_pairprod * nthreads, nthreads), [=](sycl::nd_item<3> item_ct1) {
        // process_pairprod(queues_size_ct0, block, scor, state, queues_ct4, item_ct1);
      });
    });

    // Wait for GPU to finish before accessing on host
    q_ct1.wait_and_throw();

    std::cout << "Total energy loss " << scor->totalEnergyLoss.load() << " number of secondaries "
              << scor->secondaries.load() << " blocks used " << block->GetNused() << std::endl;
  }

}
