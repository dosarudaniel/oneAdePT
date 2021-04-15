
#ifndef G4HepEmRandomEngineBase_HH
#define G4HepEmRandomEngineBase_HH

#include <CopCore/1/Ranluxpp.h>

/**
 * @file    G4HepEmRandomEngine.hh
 * @class   G4HepEmRandomEngine
 * @author  J. Hahnfeld
 * @date    2021
 *
 * A simple abstraction for a random number engine.
 *
 * Holds a reference to the real engine and two function pointers to generate
 * one random number or fill an array with a given size, respectively.
 */

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

#endif // G4HepEmRandomEngineBase_HH
