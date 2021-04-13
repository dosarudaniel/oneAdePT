
#ifndef G4HepEmRandomEngine_HH
#define G4HepEmRandomEngine_HH

#include "G4HepEmMacros.hh"
#ifdef _ONEADEPT_
#include <CopCore/1/Ranluxpp.h>
#else
#include <CopCore/Ranluxpp.h>
#endif
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
  typedef double (*FlatFn)(void *object);
  typedef void (*FlatArrayFn)(void *object, const int size, double* vect);
  
  G4HepEmHostDevice
  G4HepEmRandomEngine(void *object, FlatFn flatFn, FlatArrayFn flatArrayFn)
  : fObject(object), fFlatFn(flatFn), fFlatArrayFn(flatArrayFn) { }

  G4HepEmHostDevice
  G4HepEmRandomEngine(void *object) : fObject(object) { }

  G4HepEmHostDevice
  double flat() { return ((RanluxppDouble *)fObject)->Rndm(); }
  G4HepEmHostDevice
  void flatArray(const int size, double* vect) {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDouble *)fObject)->Rndm();
    }
  }

private:
  void *fObject;
  FlatFn fFlatFn;
  FlatArrayFn fFlatArrayFn;
};


#endif // G4HepEmRandomEngine_HH
