
#ifndef G4HepEmMath_HH
#define G4HepEmMath_HH

#include <cmath>

// #if (defined( __SYCL_DEVICE_ONLY__))
// #define log sycl::log
// #define exp sycl::exp
// #define cos sycl::cos
// #define sin sycl::sin
// #define pow sycl::pow
// #else
// #define log std::log
// #define exp std::exp
// #define cos std::cos
// #define sin std::sin
// #define pow std::pow
// #endif


#include "G4HepEmMacros.hh"

template <typename T>
G4HepEmHostDevice static inline
T G4HepEmMax(T a, T b) {
 return a > b ? a : b;
}

template <typename T>
G4HepEmHostDevice static inline
T G4HepEmMin(T a, T b) {
 return a < b ? a : b;
}

template <typename T>
G4HepEmHostDevice static inline
T G4HepEmX13(T x) {
 return pow(x, 1./3.);
}

#endif // G4HepEmMath_HH
