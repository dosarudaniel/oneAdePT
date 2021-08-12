// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0


#include <iostream>
#include <stdlib.h>

#include <CopCore/1/Ranluxpp.h>

class G4HepEmRandomEngine {
public:

  typedef double (*FlatFn)(RanluxppDouble *object);
  typedef void (*FlatArrayFn)(RanluxppDouble *object, const int size, double* vect);

  G4HepEmRandomEngine(RanluxppDouble *object, FlatFn flatFn, FlatArrayFn flatArrayFn)
    : fObject(object), fFlatFn(flatFn), fFlatArrayFn(flatArrayFn) { }

  G4HepEmRandomEngine(RanluxppDouble *object) : fObject(object) { }

  double flat() {
    return (fObject->Rndm());
  }

  void flatArray(const int size, double* vect) {
    for (int i = 0; i < size; i++) {
      vect[i] = fObject->Rndm();
    }
  }

private:
  RanluxppDouble  *fObject;
  FlatFn fFlatFn;
  FlatArrayFn fFlatArrayFn;
};

class RanluxppDoubleEngine : public G4HepEmRandomEngine {
public:
  RanluxppDoubleEngine(RanluxppDouble *engine)
    : G4HepEmRandomEngine(/*object=*/engine)
  {}
};


double SampleCostModifiedTsai(G4HepEmRandomEngine* rnge) { 
  return(rnge->flat());
};

void kernel(double *res)
{
  RanluxppDouble rl;

  double r1 = rl.Rndm();

  RanluxppDoubleEngine r(&rl);

  double r2 = r.flat();
  
  double r3 = SampleCostModifiedTsai(&r);

  *res = r1+r2+r3;
}

int main(void)
{
  sycl::default_selector device_selector;

  sycl::queue q_ct1(device_selector);
  std::cout <<  "Running on "
	    << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
	    << "\n";

  RanluxppDouble rl;

  double r1 = rl.Rndm();

  RanluxppDoubleEngine r(&rl);
  
  double r2  = r.flat();

  double r3 = SampleCostModifiedTsai(&r);

  std::cout << "host=" << r1+r2+r3 << "\n";

  double *res_dev = sycl::malloc_device<double>(1, q_ct1);

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       kernel(res_dev);
                     });
  }).wait();

  double res;

  q_ct1.memcpy(&res, res_dev, sizeof(res)).wait();
  
  std::cout << "dev=" << res << "\n";
  
}
