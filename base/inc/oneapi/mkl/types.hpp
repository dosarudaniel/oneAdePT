/*******************************************************************************
* Copyright 2018-2020 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef _TYPES_HPP__
#define _TYPES_HPP__

#include <cstdint>
#include "mkl_types.h"
#include "mkl_cblas.h"
#include "oneapi/mkl/bfloat16.hpp"

namespace oneapi {
namespace mkl {


// BLAS flag types.
enum class transpose : char {
    nontrans = 0,
    trans = 1,
    conjtrans = 3,
    N = 0,
    T = 1,
    C = 3
};

enum class uplo : char {
    upper = 0,
    lower = 1,
    U = 0,
    L = 1
};

enum class diag : char {
    nonunit = 0,
    unit = 1,
    N = 0,
    U = 1
};

enum class side : char {
    left = 0,
    right = 1,
    L = 0,
    R = 1
};

enum class offset : char {
    row = 0,
    column = 1,
    fix = 2,
    R = 0,
    C = 1,
    F = 2
};

// LAPACK flag types.
enum class job : char {
    novec = 0,
    vec = 1,
    updatevec = 2,
    allvec = 3,
    somevec = 4,
    overwritevec = 5,
    N = 0,
    V = 1,
    U = 2,
    A = 3,
    S = 4,
    O = 5
};

enum class generate : char {
    q = 0,
    p = 1,
    none = 2,
    both = 3,
    Q = 0,
    P = 1,
    N = 2,
    V = 3
};

enum class compz : char {
    novectors = 0,
    vectors = 1,
    initvectors = 2,
    N = 0,
    V = 1,
    I = 2,
};

enum class direct : char {
    forward = 0,
    backward = 1,
    F = 0,
    B = 1,
};

enum class storev : char {
    columnwise = 0,
    rowwise = 1,
    C = 0,
    R = 1,
};

enum class rangev : char {
    all = 0,
    values = 1,
    indices = 2,
    A = 0,
    V = 1,
    I = 2,
};

enum class order : char {
    block = 0,
    entire = 1,
    B = 0,
    E = 1,
};

enum class jobsvd : char {
    novec = 0,
    vectors = 1,
    vectorsina = 2,
    somevec = 3,
    N = 0,
    A = 1,
    O = 2,
    S = 3
};
// Conversion functions to traditional Fortran characters.
inline const char * fortran_char(transpose t) {
    if (t == transpose::nontrans)  return "N";
    if (t == transpose::trans)     return "T";
    if (t == transpose::conjtrans) return "C";
    return "N";
}

inline const char * fortran_char(offset t) {
    if (t == offset::fix)  return "F";
    if (t == offset::row)  return "R";
    if (t == offset::column)  return "C";
    return "N";
}

inline const char * fortran_char(uplo u) {
    if (u == uplo::upper) return "U";
    if (u == uplo::lower) return "L";
    return "U";
}

inline const char * fortran_char(diag d) {
    if (d == diag::nonunit) return "N";
    if (d == diag::unit)    return "U";
    return "N";
}

inline const char * fortran_char(side s) {
    if (s == side::left)  return "L";
    if (s == side::right) return "R";
    return "L";
}

// inline const CBLAS_TRANSPOSE cblas_enum(transpose t) {
//     if (t == transpose::nontrans)  return CblasNonTrans;
//     if (t == transpose::trans)     return CblasTrans;
//     if (t == transpose::conjtrans) return CblasConjTrans;
//     return "N";
// }

// inline const CBLAS_OFFSET cblas_enum(offset t) {
//     if (t == offset::fix)  return CblasFixOffset;
//     if (t == offset::row)  return CblasRowOffset;
//     if (t == offset::column)  return CblasColOffset;
//     return "N";
// }

// inline const CBLAS_UPLO cblas_enum(uplo u) {
//     if (u == uplo::upper) return CblasUpper;
//     if (u == uplo::lower) return CblasLower;
//     return "U";
// }

// inline const CBLAS_DIAG cblas_enum(diag d) {
//     if (d == diag::nonunit) return CblasNonUnit;
//     if (d == diag::unit)    return CblasUnit;
//     return "N";
// }

// inline const CBLAS_SIDE cblas_enum(side s) {
//     if (s == side::left)  return CblasLeft;
//     if (s == side::right) return CblasRight;
//     return "L";
// }

inline const char * fortran_char(job j) {
    if (j == job::novec)  return "N";
    if (j == job::vec)  return "V";
    if (j == job::updatevec)  return "U";
    if (j == job::allvec)  return "A";
    if (j == job::somevec)  return "S";
    if (j == job::overwritevec)  return "O";
    return "N";
}
inline const char * fortran_char(jobsvd j) {
    if (j == jobsvd::novec)  return "N";
    if (j == jobsvd::vectors)  return "A";
    if (j == jobsvd::vectorsina)  return "O";
    if (j == jobsvd::somevec)  return "S";
    return "N";
}

inline const char * fortran_char(generate v) {
    if (v == generate::q)  return "Q";
    if (v == generate::p)  return "P";
    if (v == generate::none)  return "N";
    if (v == generate::both)  return "B";
    return "Q";
}

inline const char * fortran_char(compz c) {
    if (c == compz::vectors) return "V";
    if (c == compz::initvectors) return "I";
    return "N";
}

inline const char * fortran_char(direct d) {
    if (d == direct::backward) return "B";
    return "F";
}

inline const char * fortran_char(storev s) {
    if (s == storev::rowwise) return "R";
    return "C";
}

inline const char * fortran_char(rangev r) {
    if (r == rangev::values) return "V";
    if (r == rangev::indices) return "I";
    return "A";
}

inline const char * fortran_char(order o) {
    if (o == order::entire) return "E";
    return "B";
}

// Conversion functions to CBLAS enums.
inline MKL_TRANSPOSE cblas_convert(transpose t) {
    if (t == transpose::nontrans)  return MKL_NOTRANS;
    if (t == transpose::trans)     return MKL_TRANS;
    if (t == transpose::conjtrans) return MKL_CONJTRANS;
    return MKL_NOTRANS;
}

inline MKL_UPLO cblas_convert(uplo u) {
    if (u == uplo::upper) return MKL_UPPER;
    if (u == uplo::lower) return MKL_LOWER;
    return MKL_UPPER;
}

inline MKL_DIAG cblas_convert(diag d) {
    if (d == diag::nonunit) return MKL_NONUNIT;
    if (d == diag::unit)    return MKL_UNIT;
    return MKL_NONUNIT;
}

inline MKL_SIDE cblas_convert(side s) {
    if (s == side::left)  return MKL_LEFT;
    if (s == side::right) return MKL_RIGHT;
    return MKL_LEFT;
}

enum class index_base : char {
    zero = 0,
    one  = 1,
};

} /* namespace mkl */
} // namespace oneapi

#endif /* _TYPES_HPP__ */
