!===============================================================================
! Copyright 2020 Intel Corporation.
!
! This software and the related documents are Intel copyrighted  materials,  and
! your use of  them is  governed by the  express license  under which  they were
! provided to you (License).  Unless the License provides otherwise, you may not
! use, modify, copy, publish, distribute,  disclose or transmit this software or
! the related documents without Intel's prior written permission.
!
! This software and the related documents  are provided as  is,  with no express
! or implied  warranties,  other  than those  that are  expressly stated  in the
! License.
!===============================================================================
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) FORTRAN interface for
!      OpenMP offload for LAPACK
!*******************************************************************************
  
   interface

      subroutine mkl_lapack_openmp_offload_cgebrd(m, n, a, lda, d, e,  &
                                                  tauq, taup, work,    &
                                                  lwork, info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      complex*8, intent(out) :: tauq(*)
      complex*8, intent(out) :: taup(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cgebrd

      subroutine mkl_lapack_openmp_offload_dgebrd(m, n, a, lda, d, e,  &
                                                  tauq, taup, work,    &
                                                  lwork, info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      double precision, intent(out) :: tauq(*)
      double precision, intent(out) :: taup(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dgebrd

      subroutine mkl_lapack_openmp_offload_sgebrd(m, n, a, lda, d, e,  &
                                                  tauq, taup, work,    &
                                                  lwork, info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      real, intent(out) :: tauq(*)
      real, intent(out) :: taup(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sgebrd

      subroutine mkl_lapack_openmp_offload_zgebrd(m, n, a, lda, d, e,  &
                                                  tauq, taup, work,    &
                                                  lwork, info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      complex*16, intent(out) :: tauq(*)
      complex*16, intent(out) :: taup(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zgebrd

      subroutine mkl_lapack_openmp_offload_cgeqrf(m, n, a, lda, tau,   &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(out) :: tau(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cgeqrf

      subroutine mkl_lapack_openmp_offload_dgeqrf(m, n, a, lda, tau,   &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: tau(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dgeqrf

      subroutine mkl_lapack_openmp_offload_sgeqrf(m, n, a, lda, tau,   &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: tau(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sgeqrf

      subroutine mkl_lapack_openmp_offload_zgeqrf(m, n, a, lda, tau,   &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(out) :: tau(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zgeqrf

      subroutine mkl_lapack_openmp_offload_cgesvd(jobu, jobvt, m, n, a,&
                                                  lda, s, u, ldu, vt,  &
                                                  ldvt, work, lwork,   &
                                                  rwork, info) BIND(C)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: s(*)
      complex*8, intent(out) :: u(ldu,*)
      integer, intent(in) :: ldu
      complex*8, intent(out) :: vt(ldvt,*)
      integer, intent(in) :: ldvt
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      real, intent(out) :: rwork(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cgesvd

      subroutine mkl_lapack_openmp_offload_dgesvd(jobu, jobvt, m, n, a,&
                                                  lda, s, u, ldu, vt,  &
                                                  ldvt, work, lwork,   &
                                                  info) BIND(C)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: s(*)
      double precision, intent(out) :: u(ldu,*)
      integer, intent(in) :: ldu
      double precision, intent(out) :: vt(ldvt,*)
      integer, intent(in) :: ldvt
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dgesvd

      subroutine mkl_lapack_openmp_offload_sgesvd(jobu, jobvt, m, n, a,&
                                                  lda, s, u, ldu, vt,  &
                                                  ldvt, work, lwork,   &
                                                  info) BIND(C)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: s(*)
      real, intent(out) :: u(ldu,*)
      integer, intent(in) :: ldu
      real, intent(out) :: vt(ldvt,*)
      integer, intent(in) :: ldvt
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sgesvd

      subroutine mkl_lapack_openmp_offload_zgesvd(jobu, jobvt, m, n, a,&
                                                  lda, s, u, ldu, vt,  &
                                                  ldvt, work, lwork,   &
                                                  rwork, info) BIND(C)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: s(*)
      complex*16, intent(out) :: u(ldu,*)
      integer, intent(in) :: ldu
      complex*16, intent(out) :: vt(ldvt,*)
      integer, intent(in) :: ldvt
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zgesvd

      subroutine mkl_lapack_openmp_offload_cgetrf(m, n, a, lda, ipiv,  &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: ipiv(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cgetrf

      subroutine mkl_lapack_openmp_offload_dgetrf(m, n, a, lda, ipiv,  &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: ipiv(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dgetrf

      subroutine mkl_lapack_openmp_offload_sgetrf(m, n, a, lda, ipiv,  &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: ipiv(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sgetrf

      subroutine mkl_lapack_openmp_offload_zgetrf(m, n, a, lda, ipiv,  &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: ipiv(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zgetrf

      subroutine mkl_lapack_openmp_offload_cgetri(n, a, lda, ipiv,     &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cgetri

      subroutine mkl_lapack_openmp_offload_dgetri(n, a, lda, ipiv,     &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dgetri

      subroutine mkl_lapack_openmp_offload_sgetri(n, a, lda, ipiv,     &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sgetri

      subroutine mkl_lapack_openmp_offload_zgetri(n, a, lda, ipiv,     &
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zgetri

      subroutine mkl_lapack_openmp_offload_cgetrs(trans, n, nrhs, a,   &
                                                  lda, ipiv, b, ldb,   &
                                                  info) BIND(C)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      complex*8, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      complex*8, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cgetrs

      subroutine mkl_lapack_openmp_offload_dgetrs(trans, n, nrhs, a,   &
                                                  lda, ipiv, b, ldb,   &
                                                  info) BIND(C)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      double precision, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      double precision, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dgetrs

      subroutine mkl_lapack_openmp_offload_sgetrs(trans, n, nrhs, a,   &
                                                  lda, ipiv, b, ldb,   &
                                                  info) BIND(C)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      real, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      real, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sgetrs

      subroutine mkl_lapack_openmp_offload_zgetrs(trans, n, nrhs, a,   &
                                                  lda, ipiv, b, ldb,   &
                                                  info) BIND(C)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      complex*16, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(in) :: ipiv(*)
      complex*16, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zgetrs

      subroutine mkl_lapack_openmp_offload_cheev(jobz, uplo, n, a, lda,&
                                                 w, work, lwork, rwork,&
                                                 info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: w(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      real, intent(out) :: rwork(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cheev

      subroutine mkl_lapack_openmp_offload_zheev(jobz, uplo, n, a, lda,&
                                                 w, work, lwork, rwork,&
                                                 info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zheev

      subroutine mkl_lapack_openmp_offload_cheevd(jobz, uplo, n, a,    &
                                                  lda, w, work, lwork, &
                                                  rwork, lrwork, iwork,&
                                                  liwork, info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: w(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      real, intent(out) :: rwork(*)
      integer, intent(in) :: lrwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cheevd

      subroutine mkl_lapack_openmp_offload_zheevd(jobz, uplo, n, a,    &
                                                  lda, w, work, lwork, &
                                                  rwork, lrwork, iwork,&
                                                  liwork, info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      double precision, intent(out) :: rwork(*)
      integer, intent(in) :: lrwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zheevd

      subroutine mkl_lapack_openmp_offload_cheevx(jobz, range, uplo, n,&
                                                  a, lda, vl, vu, il,  &
                                                  iu, abstol, m, w, z, &
                                                  ldz, work, lwork,    &
                                                  rwork, iwork, ifail, &
                                                  info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      real, intent(out) :: w(*)
      complex*8, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      real, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cheevx

      subroutine mkl_lapack_openmp_offload_zheevx(jobz, range, uplo, n,&
                                                  a, lda, vl, vu, il,  &
                                                  iu, abstol, m, w, z, &
                                                  ldz, work, lwork,    &
                                                  rwork, iwork, ifail, &
                                                  info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zheevx

      subroutine mkl_lapack_openmp_offload_chegvd(itype, jobz, uplo, n,&
                                                  a, lda, b, ldb, w,   &
                                                  work, lwork, rwork,  &
                                                  lrwork, iwork,       &
                                                  liwork, info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      real, intent(out) :: w(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      real, intent(out) :: rwork(*)
      integer, intent(in) :: lrwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_chegvd

      subroutine mkl_lapack_openmp_offload_zhegvd(itype, jobz, uplo, n,&
                                                  a, lda, b, ldb, w,   &
                                                  work, lwork, rwork,  &
                                                  lrwork, iwork,       &
                                                  liwork, info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      double precision, intent(out) :: rwork(*)
      integer, intent(in) :: lrwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zhegvd

      subroutine mkl_lapack_openmp_offload_chegvx(itype, jobz, range,  &
                                                  uplo, n, a, lda, b,  &
                                                  ldb, vl, vu, il, iu, &
                                                  abstol, m, w, z, ldz,&
                                                  work, lwork, rwork,  &
                                                  iwork, ifail,        &
                                                  info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      real, intent(out) :: w(*)
      complex*8, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      real, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_chegvx

      subroutine mkl_lapack_openmp_offload_zhegvx(itype, jobz, range,  &
                                                  uplo, n, a, lda, b,  &
                                                  ldb, vl, vu, il, iu, &
                                                  abstol, m, w, z, ldz,&
                                                  work, lwork, rwork,  &
                                                  iwork, ifail,        &
                                                  info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zhegvx

      subroutine mkl_lapack_openmp_offload_chetrd(uplo, n, a, lda, d,  &
                                                  e, tau, work, lwork, &
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      complex*8, intent(out) :: tau(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_chetrd

      subroutine mkl_lapack_openmp_offload_zhetrd(uplo, n, a, lda, d,  &
                                                  e, tau, work, lwork, &
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      complex*16, intent(out) :: tau(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zhetrd

      subroutine mkl_lapack_openmp_offload_dorgqr(m, n, k, a, lda, tau,&
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(in) :: tau(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dorgqr

      subroutine mkl_lapack_openmp_offload_sorgqr(m, n, k, a, lda, tau,&
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(in) :: tau(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sorgqr

      subroutine mkl_lapack_openmp_offload_dormqr(side, trans, m, n, k,&
                                                  a, lda, tau, c, ldc, &
                                                  work, lwork,         &
                                                  info) BIND(C)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      double precision, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(in) :: tau(*)
      double precision, intent(inout) :: c(ldc,*)
      integer, intent(in) :: ldc
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dormqr

      subroutine mkl_lapack_openmp_offload_sormqr(side, trans, m, n, k,&
                                                  a, lda, tau, c, ldc, &
                                                  work, lwork,         &
                                                  info) BIND(C)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      real, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(in) :: tau(*)
      real, intent(inout) :: c(ldc,*)
      integer, intent(in) :: ldc
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_sormqr

      subroutine mkl_lapack_openmp_offload_csteqr(compz, n, d, e, z,   &
                                                  ldz, work,           &
                                                  info) BIND(C)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      real, intent(inout) :: d(*)
      real, intent(inout) :: e(*)
      complex*8, intent(inout) :: z(ldz,*)
      integer, intent(in) :: ldz
      real, intent(out) :: work(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_csteqr

      subroutine mkl_lapack_openmp_offload_dsteqr(compz, n, d, e, z,   &
                                                  ldz, work,           &
                                                  info) BIND(C)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      double precision, intent(inout) :: d(*)
      double precision, intent(inout) :: e(*)
      double precision, intent(inout) :: z(ldz,*)
      integer, intent(in) :: ldz
      double precision, intent(out) :: work(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsteqr

      subroutine mkl_lapack_openmp_offload_ssteqr(compz, n, d, e, z,   &
                                                  ldz, work,           &
                                                  info) BIND(C)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      real, intent(inout) :: d(*)
      real, intent(inout) :: e(*)
      real, intent(inout) :: z(ldz,*)
      integer, intent(in) :: ldz
      real, intent(out) :: work(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssteqr

      subroutine mkl_lapack_openmp_offload_zsteqr(compz, n, d, e, z,   &
                                                  ldz, work,           &
                                                  info) BIND(C)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      double precision, intent(inout) :: d(*)
      double precision, intent(inout) :: e(*)
      complex*16, intent(inout) :: z(ldz,*)
      integer, intent(in) :: ldz
      double precision, intent(out) :: work(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zsteqr

      subroutine mkl_lapack_openmp_offload_dsyev(jobz, uplo, n, a, lda,&
                                                 w, work, lwork,       &
                                                 info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsyev

      subroutine mkl_lapack_openmp_offload_ssyev(jobz, uplo, n, a, lda,&
                                                 w, work, lwork,       &
                                                 info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: w(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssyev

      subroutine mkl_lapack_openmp_offload_dsyevd(jobz, uplo, n, a,    &
                                                  lda, w, work, lwork, &
                                                  iwork, liwork,       &
                                                  info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsyevd

      subroutine mkl_lapack_openmp_offload_ssyevd(jobz, uplo, n, a,    &
                                                  lda, w, work, lwork, &
                                                  iwork, liwork,       &
                                                  info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: w(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssyevd

      subroutine mkl_lapack_openmp_offload_dsyevx(jobz, range, uplo, n,&
                                                  a, lda, vl, vu, il,  &
                                                  iu, abstol, m, w, z, &
                                                  ldz, work, lwork,    &
                                                  iwork, ifail,        &
                                                  info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsyevx

      subroutine mkl_lapack_openmp_offload_ssyevx(jobz, range, uplo, n,&
                                                  a, lda, vl, vu, il,  &
                                                  iu, abstol, m, w, z, &
                                                  ldz, work, lwork,    &
                                                  iwork, ifail,        &
                                                  info) BIND(C)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      real, intent(out) :: w(*)
      real, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssyevx

      subroutine mkl_lapack_openmp_offload_dsygvd(itype, jobz, uplo, n,&
                                                  a, lda, b, ldb, w,   &
                                                  work, lwork, iwork,  &
                                                  liwork, info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsygvd

      subroutine mkl_lapack_openmp_offload_ssygvd(itype, jobz, uplo, n,&
                                                  a, lda, b, ldb, w,   &
                                                  work, lwork, iwork,  &
                                                  liwork, info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      real, intent(out) :: w(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssygvd

      subroutine mkl_lapack_openmp_offload_dsygvx(itype, jobz, range,  &
                                                  uplo, n, a, lda, b,  &
                                                  ldb, vl, vu, il, iu, &
                                                  abstol, m, w, z, ldz,&
                                                  work, lwork, iwork,  &
                                                  ifail, info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsygvx

      subroutine mkl_lapack_openmp_offload_ssygvx(itype, jobz, range,  &
                                                  uplo, n, a, lda, b,  &
                                                  ldb, vl, vu, il, iu, &
                                                  abstol, m, w, z, ldz,&
                                                  work, lwork, iwork,  &
                                                  ifail, info) BIND(C)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      real, intent(out) :: w(*)
      real, intent(out) :: z(ldz,*)
      integer, intent(in) :: ldz
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssygvx

      subroutine mkl_lapack_openmp_offload_dsytrd(uplo, n, a, lda, d,  &
                                                  e, tau, work, lwork, &
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      double precision, intent(out) :: tau(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dsytrd

      subroutine mkl_lapack_openmp_offload_ssytrd(uplo, n, a, lda, d,  &
                                                  e, tau, work, lwork, &
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      real, intent(out) :: tau(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ssytrd

      subroutine mkl_lapack_openmp_offload_ctrtrs(uplo, trans, diag, n,&
                                                  nrhs, a, lda, b, ldb,&
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      complex*8, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ctrtrs

      subroutine mkl_lapack_openmp_offload_dtrtrs(uplo, trans, diag, n,&
                                                  nrhs, a, lda, b, ldb,&
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      double precision, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_dtrtrs

      subroutine mkl_lapack_openmp_offload_strtrs(uplo, trans, diag, n,&
                                                  nrhs, a, lda, b, ldb,&
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      real, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_strtrs

      subroutine mkl_lapack_openmp_offload_ztrtrs(uplo, trans, diag, n,&
                                                  nrhs, a, lda, b, ldb,&
                                                  info) BIND(C)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      complex*16, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(inout) :: b(ldb,*)
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_ztrtrs

      subroutine mkl_lapack_openmp_offload_cungqr(m, n, k, a, lda, tau,&
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(in) :: tau(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cungqr

      subroutine mkl_lapack_openmp_offload_zungqr(m, n, k, a, lda, tau,&
                                                  work, lwork,         &
                                                  info) BIND(C)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(in) :: tau(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zungqr

      subroutine mkl_lapack_openmp_offload_cunmqr(side, trans, m, n, k,&
                                                  a, lda, tau, c, ldc, &
                                                  work, lwork,         &
                                                  info) BIND(C)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      complex*8, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(in) :: tau(*)
      complex*8, intent(inout) :: c(ldc,*)
      integer, intent(in) :: ldc
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_cunmqr

      subroutine mkl_lapack_openmp_offload_zunmqr(side, trans, m, n, k,&
                                                  a, lda, tau, c, ldc, &
                                                  work, lwork,         &
                                                  info) BIND(C)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      complex*16, intent(in) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(in) :: tau(*)
      complex*16, intent(inout) :: c(ldc,*)
      integer, intent(in) :: ldc
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      end subroutine mkl_lapack_openmp_offload_zunmqr

   end interface
