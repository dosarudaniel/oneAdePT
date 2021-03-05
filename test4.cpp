// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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
// queues[2]->size(), block, state, queues[2], item_ct1)

void transport(int n, adept::BlockData<MyTrack> *block, oneapi::mkl::rng::device::philox4x32x10<1> *states,
               Queue_t *queues, sycl::nd_item<1> item_ct1)
{
  for (int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
           particle < n;
           particle += item_ct1.get_local_range().get(0) * item_ct1.get_group_range(0)) {
    // transport particles
    for (int xyz = 0; xyz < 3; xyz++) {
      (*block)[particle].pos[xyz] = (*block)[particle].pos[xyz] + (*block)[particle].energy * (*block)[particle].dir[xyz];
    }
  }
}

// block, scor, state, queues, item_ct1
void select_process(adept::BlockData<MyTrack> *block, Scoring *scor, oneapi::mkl::rng::device::philox4x32x10<1> *states,
                    Queue_t *queues[], sycl::nd_item<1> item_ct1)
{
  oneapi::mkl::rng::device::uniform<float> distr_ct1;

  int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);

  // check if you are not outside the used block
  if (particle > block->GetNused() + block->GetNholes()) return;

  // check if the particle is still alive (E>0)
  if ((*block)[particle].energy == 0) return;

  // generate random number
  float r = oneapi::mkl::rng::device::generate(distr_ct1, *states);

  if (r > 0.5f) {
      queues[0]->enqueue(particle);
  } else {
      queues[1]->enqueue(particle);
  }
}

// kernel function that does energy loss

void process_eloss(int n, adept::BlockData<MyTrack> *block, Scoring *scor,
                   oneapi::mkl::rng::device::philox4x32x10<1> *states, Queue_t *queue, sycl::nd_item<1> item_ct1)
{
  for (int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
           particle < n;
           particle += item_ct1.get_local_range().get(0) * item_ct1.get_group_range(0)) {

    if (!queue->dequeue(particle)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle].energy == 0) return;

    // call the 'process'

    // energy loss
    float eloss = 0.2f * (*block)[particle].energy;
    //scor->totalEnergyLoss.fetch_add(eloss < 0.001f ? (*block)[particle].energy : eloss);

    (*block)[particle].energy = (eloss < 0.001f ? 0.0f : ((*block)[particle].energy - eloss));

    // if particle dies (E=0) release the slot
    if ((*block)[particle].energy < 0.001f) block->ReleaseElement(particle);
  }
}

// kernel function that does pair production

void process_pairprod(int n, adept::BlockData<MyTrack> *block, Scoring *scor,
                      oneapi::mkl::rng::device::philox4x32x10<1> *states, Queue_t *queue, sycl::nd_item<1> item_ct1)
{
  for (int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
           particle < n;
           particle += item_ct1.get_local_range().get(0) * item_ct1.get_group_range(0)) {

    if (!queue->dequeue(particle)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle].energy < 0.0001f) return;

    // pair production
    auto secondary_track = block->NextElement();
    assert(secondary_track != nullptr && "No slot available for secondary track");

    float eloss = 0.5f * (*block)[particle].energy;
    (*block)[particle].energy -= eloss;

    secondary_track->energy = eloss;

    // increase the counter of secondaries
    scor->secondaries.fetch_add(1);

  }
}

/* this GPU kernel function is used to initialize the random states */

void init(oneapi::mkl::rng::device::philox4x32x10<1> *states)
{
  /* we have to initialize the state */
  *states = oneapi::mkl::rng::device::philox4x32x10<1>(0, {0, 0 * 8});
}


int main()
{
  
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  oneapi::mkl::rng::device::philox4x32x10<1> *state;

  state = sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(1, q_ct1);
 
  q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range(sycl::range(1), sycl::range(1)),
 	           [=](sycl::nd_item<> item_ct1) {
		       init(state);
       });
   });

  q_ct1.wait_and_throw();

  // Capacity of the different containers
  constexpr int capacity = 1 << 20;

  using Queue_t = adept::mpmc_bounded_queue<int>;

  constexpr int numberOfProcesses = 3;

  Queue_t **queues = (Queue_t**) sycl::malloc_shared<Queue_t *>(numberOfProcesses, q_ct1);
  char *buffer[numberOfProcesses];

  size_t buffersize = Queue_t::SizeOfInstance(capacity);
  for (int i = 0; i < numberOfProcesses; i++) {
    buffer[i] = (char*) sycl::malloc_shared(buffersize, q_ct1); 
    queues[i] = Queue_t::MakeInstanceAt(capacity, buffer[i]);
  }

  // Allocate the content of Scoring in a buffer
  //Scoring *scor = new Scoring();
  char *buffer2 = (char*) sycl::malloc_shared(sizeof(Scoring), q_ct1);

  Scoring *scor = (Scoring*) Scoring::MakeInstanceAt(buffer2);



  // Initialize scoring
  scor->secondaries = 0;
  scor->totalEnergyLoss = 0;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block

  using Block_t    = adept::BlockData<MyTrack>;

  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer3 = (char*) sycl::malloc_shared(blocksize, q_ct1);
  auto block = Block_t::MakeInstanceAt(capacity, buffer3);

  // auto block = Block_t::MakeInstance(capacity);
  
  // initializing one track in the block
  auto track    = block->NextElement();

  track->energy = 100.0f;

  // initializing second track in the block
  auto track2    = block->NextElement();

  track2->energy = 30.0f;

  q_ct1.wait_and_throw();

  const sycl::range nthreads(1);
  const sycl::range maxBlocks(1);

  sycl::range numBlocks(1), numBlocks_eloss(1), numBlocks_pairprod(1), numBlocks_transport(1);
  
  while (block->GetNused()) {

    numBlocks = (block->GetNused() + block->GetNholes() + nthreads - 1) / nthreads;

    // here set the maximum number of blocks

    numBlocks_transport = std::min(numBlocks[0], maxBlocks[0]);

    q_ct1.submit([&](sycl::handler &cgh) {
	 cgh.parallel_for(sycl::nd_range<1>(numBlocks_transport * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
             transport(queues[2]->size(), block, state, queues[2], item_ct1);
         });
    });

    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(numBlocks * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
                select_process(block, scor, state, queues, item_ct1);
        });
    });

    q_ct1.wait_and_throw(); // tries to move data  host <-> device ?

    // // call the process kernels

    numBlocks_eloss    = std::min((queues[0]->size() + nthreads[0] - 1) / nthreads[0], maxBlocks[0]);

    q_ct1.submit([&](sycl::handler &cgh) {
       cgh.parallel_for(sycl::nd_range<1>(numBlocks_eloss * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
	   process_eloss(queues[0]->size(), block, scor, state, queues[0], item_ct1);
       });
    });
   
    numBlocks_pairprod = std::min((queues[1]->size() + nthreads[0] - 1) / nthreads[0], maxBlocks[0]);

    q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(numBlocks_pairprod * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
           process_pairprod(queues[1]->size(), block, scor, state, queues[1], item_ct1);
        });
    });

    // Wait for GPU to finish before accessing on host
    q_ct1.wait_and_throw();

    std::cout << "Total energy loss " << scor->totalEnergyLoss.load() << " number of secondaries "
              << scor->secondaries.load() << " blocks used " << block->GetNused() << std::endl;
  }
}
