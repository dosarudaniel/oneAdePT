// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <iostream>

#include <AdePT/1/BlockData.h>

using Queue_t = adept::mpmc_bounded_queue<int>;
using Rnd_t = oneapi::mkl::rng::device::philox4x32x10<1>;

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

void transport(int n, adept::BlockData<MyTrack> *block, Rnd_t *states,
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

void select_process(int n, adept::BlockData<MyTrack> *block, Scoring *scor, Rnd_t *states,
                    Queue_t *queues[], sycl::nd_item<1> item_ct1)
{
  oneapi::mkl::rng::device::uniform<> distr;
   
  int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
 
  //if (particle >= n) return;

  // check if you are not outside the used block
  if (particle > block->GetNused() + block->GetNholes()) return;

  // check if the particle is still alive (E>0)
  if ((*block)[particle].energy == 0) return;
 
  // generate random number
  float r = oneapi::mkl::rng::device::generate(distr, states[particle]);

  if (r > 0.5f) {
      queues[0]->enqueue(particle);
  } else {
      queues[1]->enqueue(particle);
  }
}

// kernel function that does energy loss

void process_eloss(int n, adept::BlockData<MyTrack> *block, Scoring *scor,
                   Rnd_t *states, Queue_t *queue, sycl::nd_item<1> item_ct1)
{

  for (int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
           particle < n;
           particle += item_ct1.get_local_range().get(0) * item_ct1.get_group_range(0)) {
  
    if (!queue->dequeue(particle)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle].energy == 0.0f ) return;

    // call the 'process'
    // energy loss
    
    float eloss = 0.2f * (*block)[particle].energy;
    scor->totalEnergyLoss.fetch_add(eloss < 0.001f ? (*block)[particle].energy : eloss);

    (*block)[particle].energy = (eloss < 0.001f ? 0.0f : ((*block)[particle].energy - eloss));

    // if particle dies (E=0) release the slot
    if ((*block)[particle].energy == 0.0f ) block->ReleaseElement(particle);

  }
}

// kernel function that does pair production

void process_pairprod(int n, adept::BlockData<MyTrack> *block, Scoring *scor,
                      Rnd_t *states, Queue_t *queue, sycl::nd_item<1> item_ct1)
{
  
    int particle = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);

    if (!queue->dequeue(particle)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle].energy < 0.0001f ) return;

    // pair production
    auto secondary_track = block->NextElement();
    assert(secondary_track != nullptr && "No slot available for secondary track");

    float eloss = 0.5f * (*block)[particle].energy;
    (*block)[particle].energy -= eloss;

    secondary_track->energy = eloss;

    // increase the counter of secondaries
    scor->secondaries.fetch_add(1);

}

/* this GPU kernel function is used to initialize the random states */
void init(int seed, Rnd_t *states,sycl::item<1> item )
{
  // we have to initialize the state
  int id =  item.get_id(0);
  //Rnd_t engine(seed, id );
  Rnd_t engine(seed);
  oneapi::mkl::rng::device::skip_ahead(engine, id);
  states[id] = engine;
} 

int main()
{ 
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  const int seed = 1;
  
  const sycl::range nthreads(32);
  const sycl::range maxBlocks(10);

 // Capacity of the different containers
  constexpr int capacity = 1 << 20;
  constexpr int numberOfProcesses = 3;

  using Rnd_t = oneapi::mkl::rng::device::philox4x32x10<1>;
  Rnd_t* states = (Rnd_t*) sycl::malloc_device(sizeof(Rnd_t)*capacity, q_ct1);

  q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>(capacity),
 	           [=](sycl::item<> item) {
             init(seed,states,item);
       });
   }).wait();

 
  using Queue_t = adept::mpmc_bounded_queue<int>;  
  Queue_t **queues  = sycl::malloc_shared<Queue_t *>(numberOfProcesses, q_ct1);
  size_t buffersize = Queue_t::SizeOfInstance(capacity);
  char *buffer;
  for (int i = 0; i < numberOfProcesses; i++) {
    buffer = (char *)sycl::malloc_shared(buffersize, q_ct1);
    queues[i] = Queue_t::MakeInstanceAt(capacity, buffer);
  }
  
  
  // Allocate the content of Scoring in a buffer
  char *sbuffer = (char *)sycl::malloc_shared(sizeof(Scoring), q_ct1);
  Scoring *scor = Scoring::MakeInstanceAt(sbuffer);

  // Initialize scoring
  scor->secondaries = 0;
  scor->totalEnergyLoss = 0;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block

  using Block_t    = adept::BlockData<MyTrack>;

  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *bbuffer    = (char *)sycl::malloc_shared(blocksize, q_ct1);
  auto block = Block_t::MakeInstanceAt(capacity, bbuffer);
  
  // initializing one track in the block
  auto track    = block->NextElement();

  track->energy = 100.0f;

  // initializing second track in the block
  auto track2    = block->NextElement();

  track2->energy = 30.0f;

  q_ct1.wait_and_throw();

  sycl::range numBlocks(1), numBlocks_eloss(1), numBlocks_pairprod(1), numBlocks_transport(1);
  
  while (block->GetNused()>0) {

    numBlocks = (block->GetNused() + block->GetNholes() + nthreads - 1) / nthreads;

    // here set the maximum number of blocks

    numBlocks_transport = std::min(numBlocks[0], maxBlocks[0]);

    q_ct1.submit([&](sycl::handler &cgh) {
	      cgh.parallel_for(sycl::nd_range<1>(numBlocks_transport * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
             transport(queues[2]->size(), block, states, queues[2], item_ct1);
         });
    });
    
    q_ct1.submit([&](sycl::handler &cgh) {
	       cgh.parallel_for(sycl::nd_range<1>(numBlocks * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
     	      select_process(numBlocks.get(0) * nthreads.get(0), block, scor, states, queues, item_ct1);
	    });
    });//.wait();

    // call the process kernels
   
   q_ct1.wait_and_throw();

    numBlocks_eloss    = std::min((queues[0]->size() + nthreads[0] - 1) / nthreads[0] + 1, maxBlocks[0]);

    q_ct1.submit([&](sycl::handler &cgh) {
       cgh.parallel_for(sycl::nd_range<1>(numBlocks_eloss * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
      	   process_eloss(queues[0]->size(), block, scor, states, queues[0], item_ct1);
       });
    });//.wait();
    
    numBlocks_pairprod = std::min((queues[1]->size() + nthreads[0] - 1) / nthreads[0] + 1, maxBlocks[0]);

    q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(numBlocks_pairprod * nthreads, nthreads), [=](sycl::nd_item<1> item_ct1) {
           process_pairprod(queues[1]->size(), block, scor, states, queues[1], item_ct1);
        });
    });//.wait();

    // Wait for GPU to finish before accessing on host
    q_ct1.wait_and_throw();

    std::cout << "Total energy loss " << scor->totalEnergyLoss.load() << " number of secondaries "
              << scor->secondaries.load() << " blocks used " << block->GetNused() << std::endl;
  }
}
