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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) FORTRAN interface 
!      for BLAS OpenMP offload
!*******************************************************************************
  
  interface

     ! BLAS Level3

     subroutine mkl_blas_dgemm_cl_offload ( transa, transb, m, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: transa, transb
       integer,intent(in)                 :: m, n, k, lda, ldb, ldc
       double precision,intent(in)        :: alpha, beta
       double precision,intent(in)        :: a( lda, * ), b( ldb, * )
       double precision,intent(inout)     :: c( ldc, * )
     end subroutine mkl_blas_dgemm_cl_offload

     subroutine mkl_blas_sgemm_cl_offload ( transa, transb, m, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: transa, transb
       integer,intent(in)                  :: m, n, k, lda, ldb, ldc
       real,intent(in)                     :: alpha, beta
       real,intent(in)                     :: a( lda, * ), b( ldb, * )
       real,intent(in)                     :: c( ldc, * )
     end subroutine mkl_blas_sgemm_cl_offload

     subroutine mkl_blas_zgemm_cl_offload ( transa, transb, m, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: transa, transb
       integer,intent(in)                 :: m, n, k, lda, ldb, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zgemm_cl_offload

     subroutine mkl_blas_cgemm_cl_offload ( transa, transb, m, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: transa, transb
       integer,intent(in)                  :: m, n, k, lda, ldb, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_cgemm_cl_offload

     subroutine mkl_blas_dsymm_cl_offload ( side, uplo, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: side, uplo
       integer,intent(in)                 :: m, n, lda, ldb, ldc
       double precision,intent(in)        :: alpha, beta
       double precision,intent(in)        :: a( lda, * ), b( ldb, * )
       double precision,intent(inout)     :: c( ldc, * )
     end subroutine mkl_blas_dsymm_cl_offload

     subroutine mkl_blas_ssymm_cl_offload ( side, uplo, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: side, uplo
       integer,intent(in)                  :: m, n, lda, ldb, ldc
       real,intent(in)                     :: alpha, beta
       real,intent(in)                     :: a( lda, * ), b( ldb, * )
       real,intent(in)                     :: c( ldc, * )
     end subroutine mkl_blas_ssymm_cl_offload

     subroutine mkl_blas_zsymm_cl_offload ( side, uplo, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: side, uplo
       integer,intent(in)                 :: m, n, lda, ldb, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zsymm_cl_offload

     subroutine mkl_blas_csymm_cl_offload ( side, uplo, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: side, uplo
       integer,intent(in)                  :: m, n, lda, ldb, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_csymm_cl_offload

     subroutine mkl_blas_zhemm_cl_offload ( side, uplo, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: side, uplo
       integer,intent(in)                 :: m, n, lda, ldb, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zhemm_cl_offload

     subroutine mkl_blas_chemm_cl_offload ( side, uplo, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: side, uplo
       integer,intent(in)                  :: m, n, lda, ldb, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_chemm_cl_offload

     subroutine mkl_blas_dsyrk_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, trans
       integer,intent(in)                 :: n, k, lda, ldc
       double precision,intent(in)        :: alpha, beta
       double precision,intent(in)        :: a( lda, * )
       double precision,intent(inout)     :: c( ldc, * )
     end subroutine mkl_blas_dsyrk_cl_offload

     subroutine mkl_blas_ssyrk_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, trans
       integer,intent(in)                  :: n, k, lda, ldc
       real,intent(in)                     :: alpha, beta
       real,intent(in)                     :: a( lda, * )
       real,intent(in)                     :: c( ldc, * )
     end subroutine mkl_blas_ssyrk_cl_offload

     subroutine mkl_blas_zsyrk_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, trans
       integer,intent(in)                 :: n, k, lda, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zsyrk_cl_offload

     subroutine mkl_blas_csyrk_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, trans
       integer,intent(in)                  :: n, k, lda, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_csyrk_cl_offload

     subroutine mkl_blas_zherk_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, trans
       integer,intent(in)                 :: n, k, lda, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zherk_cl_offload

     subroutine mkl_blas_cherk_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, trans
       integer,intent(in)                  :: n, k, lda, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_cherk_cl_offload

     subroutine mkl_blas_dsyr2k_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, trans
       integer,intent(in)                 :: n, k, lda, ldb, ldc
       double precision,intent(in)        :: alpha, beta
       double precision,intent(in)        :: a( lda, * ), b( ldb, * )
       double precision,intent(inout)     :: c( ldc, * )
     end subroutine mkl_blas_dsyr2k_cl_offload

     subroutine mkl_blas_ssyr2k_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, trans
       integer,intent(in)                  :: n, k, lda, ldb, ldc
       real,intent(in)                     :: alpha, beta
       real,intent(in)                     :: a( lda, * ), b( ldb, * )
       real,intent(in)                     :: c( ldc, * )
     end subroutine mkl_blas_ssyr2k_cl_offload

     subroutine mkl_blas_zsyr2k_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, trans
       integer,intent(in)                 :: n, k, lda, ldb, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zsyr2k_cl_offload

     subroutine mkl_blas_csyr2k_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, trans
       integer,intent(in)                  :: n, k, lda, ldb, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_csyr2k_cl_offload

     subroutine mkl_blas_zher2k_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, trans
       integer,intent(in)                 :: n, k, lda, ldb, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zher2k_cl_offload

     subroutine mkl_blas_cher2k_cl_offload ( uplo, trans, n, k, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, trans
       integer,intent(in)                  :: n, k, lda, ldb, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_cher2k_cl_offload

     subroutine mkl_blas_dtrmm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)             :: side, uplo, trans, diag
       integer,intent(in)                 :: m, n, lda, ldb
       double precision,intent(in)        :: alpha
       double precision,intent(in)        :: a( lda, * )
       double precision,intent(inout)     :: b( ldb, * )
     end subroutine mkl_blas_dtrmm_cl_offload

     subroutine mkl_blas_strmm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)              :: side, uplo, trans, diag
       integer,intent(in)                  :: m, n, lda, ldb
       real,intent(in)                     :: alpha
       real,intent(in)                     :: a( lda, * )
       real,intent(in)                     :: b( ldb, * )
     end subroutine mkl_blas_strmm_cl_offload

     subroutine mkl_blas_ztrmm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)             :: side, uplo, trans, diag
       integer,intent(in)                 :: m, n, lda, ldb
       complex*16,intent(in)              :: alpha
       complex*16,intent(in)              :: a( lda, * )
       complex*16,intent(inout)           :: b( ldb, * )
     end subroutine mkl_blas_ztrmm_cl_offload

     subroutine mkl_blas_ctrmm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)              :: side, uplo, trans, diag
       integer,intent(in)                  :: m, n, lda, ldb
       complex,intent(in)                  :: alpha
       complex,intent(in)                  :: a( lda, * )
       complex,intent(in)                  :: b( ldb, * )
     end subroutine mkl_blas_ctrmm_cl_offload

     subroutine mkl_blas_dtrsm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)             :: side, uplo, trans, diag
       integer,intent(in)                 :: m, n, lda, ldb
       double precision,intent(in)        :: alpha
       double precision,intent(in)        :: a( lda, * )
       double precision,intent(inout)     :: b( ldb, * )
     end subroutine mkl_blas_dtrsm_cl_offload

     subroutine mkl_blas_strsm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)              :: side, uplo, trans, diag
       integer,intent(in)                  :: m, n, lda, ldb
       real,intent(in)                     :: alpha
       real,intent(in)                     :: a( lda, * )
       real,intent(in)                     :: b( ldb, * )
     end subroutine mkl_blas_strsm_cl_offload

     subroutine mkl_blas_ztrsm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)             :: side, uplo, trans, diag
       integer,intent(in)                 :: m, n, lda, ldb
       complex*16,intent(in)              :: alpha
       complex*16,intent(in)              :: a( lda, * )
       complex*16,intent(inout)           :: b( ldb, * )
     end subroutine mkl_blas_ztrsm_cl_offload

     subroutine mkl_blas_ctrsm_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda,        &
          &b, ldb ) BIND(C)
       character*1,intent(in)              :: side, uplo, trans, diag
       integer,intent(in)                  :: m, n, lda, ldb
       complex,intent(in)                  :: alpha
       complex,intent(in)                  :: a( lda, * )
       complex,intent(in)                  :: b( ldb, * )
     end subroutine mkl_blas_ctrsm_cl_offload

     subroutine mkl_blas_dgemmt_cl_offload ( uplo, transa, transb, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, transa, transb
       integer,intent(in)                 :: m, n, lda, ldb, ldc
       double precision,intent(in)        :: alpha, beta
       double precision,intent(in)        :: a( lda, * ), b( ldb, * )
       double precision,intent(inout)     :: c( ldc, * )
     end subroutine mkl_blas_dgemmt_cl_offload

     subroutine mkl_blas_sgemmt_cl_offload ( uplo, transa, transb, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, transa, transb
       integer,intent(in)                  :: m, n, lda, ldb, ldc
       real,intent(in)                     :: alpha, beta
       real,intent(in)                     :: a( lda, * ), b( ldb, * )
       real,intent(in)                     :: c( ldc, * )
     end subroutine mkl_blas_sgemmt_cl_offload

     subroutine mkl_blas_zgemmt_cl_offload ( uplo, transa, transb, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)             :: uplo, transa, transb
       integer,intent(in)                 :: m, n, lda, ldb, ldc
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zgemmt_cl_offload

     subroutine mkl_blas_cgemmt_cl_offload ( uplo, transa, transb, m, n, alpha, a, lda,        &
          &b, ldb, beta, c, ldc ) BIND(C)
       character*1,intent(in)              :: uplo, transa, transb
       integer,intent(in)                  :: m, n, lda, ldb, ldc
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_cgemmt_cl_offload

     ! BLAS Level2

     subroutine mkl_blas_dgbmv_cl_offload ( trans, m, n, kl, ku, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       double precision,intent(in)         :: alpha, beta
       integer,intent(in)                  :: incx, incy, kl, ku, lda, m, n
       character*1,intent(in)              :: trans
       double precision,intent(in)         :: a( lda, * ), x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_dgbmv_cl_offload

     subroutine mkl_blas_sgbmv_cl_offload ( trans, m, n, kl, ku, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       real,intent(in)                     :: alpha, beta
       integer,intent(in)                  :: incx, incy, kl, ku, lda, m, n
       character*1,intent(in)              :: trans
       real,intent(in)                     :: a( lda, * ), x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_sgbmv_cl_offload

     subroutine mkl_blas_zgbmv_cl_offload ( trans, m, n, kl, ku, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex*16,intent(in)               :: alpha, beta
       integer,intent(in)                  :: incx, incy, kl, ku, lda, m, n
       character*1,intent(in)              :: trans
       complex*16,intent(in)               :: a( lda, * ), x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zgbmv_cl_offload

     subroutine mkl_blas_cgbmv_cl_offload ( trans, m, n, kl, ku, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex,intent(in)                  :: alpha, beta
       integer,intent(in)                  :: incx, incy, kl, ku, lda, m, n
       character*1,intent(in)              :: trans
       complex,intent(in)                  :: a( lda, * ), x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_cgbmv_cl_offload

     subroutine mkl_blas_dgemv_cl_offload ( trans, m, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       double precision,intent(in)         :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, m, n
       character*1,intent(in)              :: trans
       double precision,intent(in)         :: a( lda, * ), x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_dgemv_cl_offload

     subroutine mkl_blas_sgemv_cl_offload ( trans, m, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       real,intent(in)                     :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, m, n
       character*1,intent(in)              :: trans
       real,intent(in)                     :: a( lda, * ), x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_sgemv_cl_offload

     subroutine mkl_blas_zgemv_cl_offload ( trans, m, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex*16,intent(in)               :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, m, n
       character*1,intent(in)              :: trans
       complex*16,intent(in)               :: a( lda, * ), x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zgemv_cl_offload

     subroutine mkl_blas_cgemv_cl_offload ( trans, m, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex,intent(in)                  :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, m, n
       character*1,intent(in)              :: trans
       complex,intent(in)                  :: a( lda, * ), x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_cgemv_cl_offload

     subroutine mkl_blas_dsbmv_cl_offload ( uplo, n, k, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       double precision,intent(in)         :: alpha, beta
       integer,intent(in)                  :: incx, incy, k, lda, n
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: a( lda, * ), x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_dsbmv_cl_offload

     subroutine mkl_blas_ssbmv_cl_offload ( uplo, n, k, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       real,intent(in)                     :: alpha, beta
       integer,intent(in)                  :: incx, incy, k, lda, n
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: a( lda, * ), x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_ssbmv_cl_offload

     subroutine mkl_blas_zhbmv_cl_offload ( uplo, n, k, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex*16,intent(in)               :: alpha, beta
       integer,intent(in)                  :: incx, incy, k, lda, n
       character*1,intent(in)              :: uplo
       complex*16,intent(in)               :: a( lda, * ), x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zhbmv_cl_offload

     subroutine mkl_blas_chbmv_cl_offload ( uplo, n, k, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex,intent(in)                  :: alpha, beta
       integer,intent(in)                  :: incx, incy, k, lda, n
       character*1,intent(in)              :: uplo
       complex,intent(in)                  :: a( lda, * ), x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_chbmv_cl_offload

     subroutine mkl_blas_dsymv_cl_offload ( uplo, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       double precision,intent(in)         :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, n
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: a( lda, * ), x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_dsymv_cl_offload

     subroutine mkl_blas_ssymv_cl_offload ( uplo, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       real,intent(in)                     :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, n
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: a( lda, * ), x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_ssymv_cl_offload

     subroutine mkl_blas_zhemv_cl_offload ( uplo, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex*16,intent(in)               :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, n
       character*1,intent(in)              :: uplo
       complex*16,intent(in)               :: a( lda, * ), x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zhemv_cl_offload

     subroutine mkl_blas_chemv_cl_offload ( uplo, n, alpha, a, lda, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex,intent(in)                  :: alpha, beta
       integer,intent(in)                  :: incx, incy, lda, n
       character*1,intent(in)              :: uplo
       complex,intent(in)                  :: a( lda, * ), x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_chemv_cl_offload

     subroutine mkl_blas_dspmv_cl_offload ( uplo, n, alpha, a, x, incx,   &
          &beta, y, incy ) BIND(C)
       double precision,intent(in)         :: alpha, beta
       integer,intent(in)                  :: incx, incy, n
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: a( * ), x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_dspmv_cl_offload

     subroutine mkl_blas_sspmv_cl_offload ( uplo, n, alpha, a, x, incx,   &
          &beta, y, incy ) BIND(C)
       real,intent(in)                     :: alpha, beta
       integer,intent(in)                  :: incx, incy, n
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: a( * ), x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_sspmv_cl_offload

     subroutine mkl_blas_zhpmv_cl_offload ( uplo, n, alpha, a, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex*16,intent(in)               :: alpha, beta
       integer,intent(in)                  :: incx, incy, n
       character*1,intent(in)              :: uplo
       complex*16,intent(in)               :: a( * ), x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zhpmv_cl_offload

     subroutine mkl_blas_chpmv_cl_offload ( uplo, n, alpha, a, x, incx,   &
          &beta, y, incy ) BIND(C)
       complex,intent(in)                  :: alpha, beta
       integer,intent(in)                  :: incx, incy, n
       character*1,intent(in)              :: uplo
       complex,intent(in)                  :: a( * ), x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_chpmv_cl_offload

     subroutine mkl_blas_dger_cl_offload ( m, n, alpha, x, incx,   &
          &y, incy, a, lda ) BIND(C)
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, incy, lda, m, n
       double precision,intent(in)         :: y( * ), x( * )
       double precision,intent(inout)      :: a( lda, * )
     end subroutine mkl_blas_dger_cl_offload

     subroutine mkl_blas_sger_cl_offload ( m, n, alpha, x, incx,   &
          &y, incy, a, lda ) BIND(C)
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, incy, lda, m, n
       real,intent(in)                     :: y( * ), x( * )
       real,intent(inout)                  :: a( lda, * )
     end subroutine mkl_blas_sger_cl_offload

     subroutine mkl_blas_zgerc_cl_offload ( m, n, alpha, x, incx,   &
          &y, incy, a, lda ) BIND(C)
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, incy, lda, m, n
       complex*16,intent(in)               :: y( * ), x( * )
       complex*16,intent(inout)            :: a( lda, * )
     end subroutine mkl_blas_zgerc_cl_offload

     subroutine mkl_blas_cgerc_cl_offload ( m, n, alpha, x, incx,   &
          &y, incy, a, lda ) BIND(C)
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, incy, lda, m, n
       complex,intent(in)                  :: y( * ), x( * )
       complex,intent(inout)               :: a( lda, * )
     end subroutine mkl_blas_cgerc_cl_offload

     subroutine mkl_blas_zgeru_cl_offload ( m, n, alpha, x, incx,   &
          &y, incy, a, lda ) BIND(C)
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, incy, lda, m, n
       complex*16,intent(in)               :: y( * ), x( * )
       complex*16,intent(inout)            :: a( lda, * )
     end subroutine mkl_blas_zgeru_cl_offload

     subroutine mkl_blas_cgeru_cl_offload ( m, n, alpha, x, incx,   &
          &y, incy, a, lda ) BIND(C)
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, incy, lda, m, n
       complex,intent(in)                  :: y( * ), x( * )
       complex,intent(inout)               :: a( lda, * )
     end subroutine mkl_blas_cgeru_cl_offload

     subroutine mkl_blas_dsyr_cl_offload ( uplo, n, alpha, x, incx,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, lda, n
       double precision,intent(in)         :: x( * )
       double precision,intent(inout)      :: a( lda, * )
     end subroutine mkl_blas_dsyr_cl_offload

     subroutine mkl_blas_ssyr_cl_offload ( uplo, n, alpha, x, incx,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, lda, n
       real,intent(in)                     :: x( * )
       real,intent(inout)                  :: a( lda, * )
     end subroutine mkl_blas_ssyr_cl_offload

     subroutine mkl_blas_zher_cl_offload ( uplo, n, alpha, x, incx,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, lda, n
       complex*16,intent(in)               :: x( * )
       complex*16,intent(inout)            :: a( lda, * )
     end subroutine mkl_blas_zher_cl_offload

     subroutine mkl_blas_cher_cl_offload ( uplo, n, alpha, x, incx,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, lda, n
       complex,intent(in)                  :: x( * )
       complex,intent(inout)               :: a( lda, * )
     end subroutine mkl_blas_cher_cl_offload

     subroutine mkl_blas_dspr_cl_offload ( uplo, n, alpha, x, incx,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, n
       double precision,intent(in)         :: x( * )
       double precision,intent(inout)      :: a( * )
     end subroutine mkl_blas_dspr_cl_offload

     subroutine mkl_blas_sspr_cl_offload ( uplo, n, alpha, x, incx,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, n
       real,intent(in)                     :: x( * )
       real,intent(inout)                  :: a( * )
     end subroutine mkl_blas_sspr_cl_offload

     subroutine mkl_blas_zhpr_cl_offload ( uplo, n, alpha, x, incx,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, n
       complex*16,intent(in)               :: x( * )
       complex*16,intent(inout)            :: a( * )
     end subroutine mkl_blas_zhpr_cl_offload

     subroutine mkl_blas_chpr_cl_offload ( uplo, n, alpha, x, incx,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       real,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, n
       complex,intent(in)                  :: x( * )
       complex,intent(inout)               :: a( * )
     end subroutine mkl_blas_chpr_cl_offload

     subroutine mkl_blas_dsyr2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, incy, lda, n
       double precision,intent(in)         :: x( * ), y( * )
       double precision,intent(inout)      :: a( lda, * )
     end subroutine mkl_blas_dsyr2_cl_offload

     subroutine mkl_blas_ssyr2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, incy, lda, n
       real,intent(in)                     :: x( * ), y( * )
       real,intent(inout)                  :: a( lda, * )
     end subroutine mkl_blas_ssyr2_cl_offload

     subroutine mkl_blas_zher2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, incy, lda, n
       complex*16,intent(in)               :: x( * ), y( * )
       complex*16,intent(inout)            :: a( lda, * )
     end subroutine mkl_blas_zher2_cl_offload

     subroutine mkl_blas_cher2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a, lda ) BIND(C)
       character*1,intent(in)              :: uplo
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, incy, lda, n
       complex,intent(in)                  :: x( * ), y( * )
       complex,intent(inout)               :: a( lda, * )
     end subroutine mkl_blas_cher2_cl_offload

     subroutine mkl_blas_dspr2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(in)         :: x( * ), y( * )
       double precision,intent(inout)      :: a( * )
     end subroutine mkl_blas_dspr2_cl_offload

     subroutine mkl_blas_sspr2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: x( * ), y( * )
       real,intent(inout)                  :: a( * )
     end subroutine mkl_blas_sspr2_cl_offload

     subroutine mkl_blas_zhpr2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(in)               :: x( * ), y( * )
       complex*16,intent(inout)            :: a( * )
     end subroutine mkl_blas_zhpr2_cl_offload

     subroutine mkl_blas_chpr2_cl_offload ( uplo, n, alpha, x, incx, y, incy,   &
          &a ) BIND(C)
       character*1,intent(in)              :: uplo
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, incy, n
       complex,intent(in)                  :: x( * ), y( * )
       complex,intent(inout)               :: a( * )
     end subroutine mkl_blas_chpr2_cl_offload

     subroutine mkl_blas_dtbmv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       double precision,intent(in)         :: a( lda, * )
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dtbmv_cl_offload

     subroutine mkl_blas_stbmv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       real,intent(in)                     :: a( lda, * )
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_stbmv_cl_offload

     subroutine mkl_blas_ztbmv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex*16,intent(in)               :: a( lda, * )
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_ztbmv_cl_offload

     subroutine mkl_blas_ctbmv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex,intent(in)                  :: a( lda, * )
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_ctbmv_cl_offload

     subroutine mkl_blas_dtrmv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       double precision,intent(in)         :: a( lda, * )
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dtrmv_cl_offload

     subroutine mkl_blas_strmv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       real,intent(in)                     :: a( lda, * )
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_strmv_cl_offload

     subroutine mkl_blas_ztrmv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex*16,intent(in)               :: a( lda, * )
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_ztrmv_cl_offload

     subroutine mkl_blas_ctrmv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex,intent(in)                  :: a( lda, * )
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_ctrmv_cl_offload

     subroutine mkl_blas_dtpmv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       double precision,intent(in)         :: a( * )
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dtpmv_cl_offload

     subroutine mkl_blas_stpmv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       real,intent(in)                     :: a( * )
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_stpmv_cl_offload

     subroutine mkl_blas_ztpmv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       complex*16,intent(in)               :: a( * )
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_ztpmv_cl_offload

     subroutine mkl_blas_ctpmv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       complex,intent(in)                  :: a( * )
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_ctpmv_cl_offload
     
     subroutine mkl_blas_dtbsv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       double precision,intent(in)         :: a( lda, * )
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dtbsv_cl_offload

     subroutine mkl_blas_stbsv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       real,intent(in)                     :: a( lda, * )
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_stbsv_cl_offload

     subroutine mkl_blas_ztbsv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex*16,intent(in)               :: a( lda, * )
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_ztbsv_cl_offload

     subroutine mkl_blas_ctbsv_cl_offload ( uplo, trans, diag, n, k, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, k, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex,intent(in)                  :: a( lda, * )
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_ctbsv_cl_offload

     subroutine mkl_blas_dtrsv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       double precision,intent(in)         :: a( lda, * )
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dtrsv_cl_offload

     subroutine mkl_blas_strsv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       real,intent(in)                     :: a( lda, * )
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_strsv_cl_offload

     subroutine mkl_blas_ztrsv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex*16,intent(in)               :: a( lda, * )
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_ztrsv_cl_offload

     subroutine mkl_blas_ctrsv_cl_offload ( uplo, trans, diag, n, a, lda, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, lda, n
       character*1,intent(in)              :: uplo, trans, diag
       complex,intent(in)                  :: a( lda, * )
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_ctrsv_cl_offload

     subroutine mkl_blas_dtpsv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       double precision,intent(in)         :: a( * )
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dtpsv_cl_offload

     subroutine mkl_blas_stpsv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       real,intent(in)                     :: a( * )
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_stpsv_cl_offload

     subroutine mkl_blas_ztpsv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       complex*16,intent(in)               :: a( * )
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_ztpsv_cl_offload

     subroutine mkl_blas_ctpsv_cl_offload ( uplo, trans, diag, n, a, x, incx ) BIND(C)
       integer,intent(in)                  :: incx, n
       character*1,intent(in)              :: uplo, trans, diag
       complex,intent(in)                  :: a( * )
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_ctpsv_cl_offload

     ! BLAS Level1

     function mkl_blas_dasum_cl_offload ( n, x, incx ) BIND(C)
       double precision                    :: dasum
       integer,intent(in)                  :: incx, n
       double precision,intent(in)         :: x( * )
     end function mkl_blas_dasum_cl_offload

     function mkl_blas_sasum_cl_offload ( n, x, incx ) BIND(C)
       real                                :: sasum
       integer,intent(in)                  :: incx, n
       real,intent(in)                     :: x( * )
     end function mkl_blas_sasum_cl_offload

     function mkl_blas_dzasum_cl_offload ( n, x, incx ) BIND(C)
       double precision                    :: dzasum
       integer,intent(in)                  :: incx, n
       complex*16,intent(in)               :: x( * )
     end function mkl_blas_dzasum_cl_offload

     function mkl_blas_scasum_cl_offload ( n, x, incx ) BIND(C)
       real                                :: scasum
       integer,intent(in)                  :: incx, n
       complex,intent(in)                  :: x( * )
     end function mkl_blas_scasum_cl_offload

     function mkl_blas_dnrm2_cl_offload ( n, x, incx ) BIND(C)
       double precision                    :: dnrm2
       integer,intent(in)                  :: incx, n
       double precision,intent(in)         :: x( * )
     end function mkl_blas_dnrm2_cl_offload

     function mkl_blas_snrm2_cl_offload ( n, x, incx ) BIND(C)
       real                                :: snrm2
       integer,intent(in)                  :: incx, n
       real,intent(in)                     :: x( * )
     end function mkl_blas_snrm2_cl_offload

     function mkl_blas_dznrm2_cl_offload ( n, x, incx ) BIND(C)
       double precision                    :: dznrm2
       integer,intent(in)                  :: incx, n
       complex*16,intent(in)               :: x( * )
     end function mkl_blas_dznrm2_cl_offload

     function mkl_blas_scnrm2_cl_offload ( n, x, incx ) BIND(C)
       real                                :: scnrm2
       integer,intent(in)                  :: incx, n
       complex,intent(in)                  :: x( * )
     end function mkl_blas_scnrm2_cl_offload

     subroutine mkl_blas_daxpy_cl_offload ( n, alpha, x, incx,   &
          &y, incy ) BIND(C)
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(in)         :: x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_daxpy_cl_offload

     subroutine mkl_blas_saxpy_cl_offload ( n, alpha, x, incx,   &
          &y, incy ) BIND(C)
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_saxpy_cl_offload

     subroutine mkl_blas_zaxpy_cl_offload ( n, alpha, x, incx,   &
          &y, incy ) BIND(C)
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(in)               :: x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zaxpy_cl_offload

     subroutine mkl_blas_caxpy_cl_offload ( n, alpha, x, incx,   &
          &y, incy ) BIND(C)
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, incy, n
       complex,intent(in)                  :: x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_caxpy_cl_offload

     subroutine mkl_blas_dcopy_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(in)         :: x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_dcopy_cl_offload

     subroutine mkl_blas_scopy_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_scopy_cl_offload

     subroutine mkl_blas_zcopy_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(in)               :: x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zcopy_cl_offload

     subroutine mkl_blas_ccopy_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       complex,intent(in)                  :: x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_ccopy_cl_offload

     function mkl_blas_ddot_cl_offload ( n, x, incx, y, incy ) BIND(C)
       double precision                    :: ddot
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(in)         :: x( * ), y( * )
     end function mkl_blas_ddot_cl_offload

     function mkl_blas_sdot_cl_offload ( n, x, incx, y, incy ) BIND(C)
       real                                :: sdot
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: x( * ), y( * )
     end function mkl_blas_sdot_cl_offload

     function mkl_blas_dsdot_cl_offload ( n, x, incx, y, incy ) BIND(C)
       double precision                    :: dsdot
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: x( * ), y( * )
     end function mkl_blas_dsdot_cl_offload

     function mkl_blas_sdsdot_cl_offload ( n, sb, x, incx, y, incy ) BIND(C)
       real                                :: sdsdot
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: sb, x( * ), y( * )
     end function mkl_blas_sdsdot_cl_offload

     function mkl_blas_zdotc_cl_offload ( n, x, incx, y, incy )
       complex*16                          :: zdotc
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(in)               :: x( * ), y( * )
     end function mkl_blas_zdotc_cl_offload

     function mkl_blas_cdotc_cl_offload ( n, x, incx, y, incy )
       complex                             :: cdotc
       integer,intent(in)                  :: incx, incy, n
       complex,intent(in)                  :: x( * ), y( * )
     end function mkl_blas_cdotc_cl_offload

     function mkl_blas_zdotu_cl_offload ( n, x, incx, y, incy )
       complex*16                          :: zdotu
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(in)               :: x( * ), y( * )
     end function mkl_blas_zdotu_cl_offload

     function mkl_blas_cdotu_cl_offload ( n, x, incx, y, incy )
       complex                             :: cdotu
       integer,intent(in)                  :: incx, incy, n
       complex,intent(in)                  :: x( * ), y( * )
     end function mkl_blas_cdotu_cl_offload

     subroutine mkl_blas_drot_cl_offload ( n, x, incx,   &
          &y, incy, c, s ) BIND(C)
       double precision,intent(in)         :: c, s         
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(inout)      :: x( * ), y( * )
     end subroutine mkl_blas_drot_cl_offload

     subroutine mkl_blas_srot_cl_offload ( n, x, incx,   &
          &y, incy, c, s ) BIND(C)
       real,intent(in)                     :: c, s         
       integer,intent(in)                  :: incx, incy, n
       real,intent(inout)                  :: x( * ), y( * )
     end subroutine mkl_blas_srot_cl_offload

     subroutine mkl_blas_zdrot_cl_offload ( n, x, incx,   &
          &y, incy, c, s ) BIND(C)
       double precision,intent(in)         :: c, s         
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(inout)            :: x( * ), y( * )
     end subroutine mkl_blas_zdrot_cl_offload

     subroutine mkl_blas_csrot_cl_offload ( n, x, incx,   &
          &y, incy, c, s ) BIND(C)
       real,intent(in)                     :: c, s         
       integer,intent(in)                  :: incx, incy, n
       complex,intent(inout)               :: x( * ), y( * )
     end subroutine mkl_blas_csrot_cl_offload

     subroutine mkl_blas_drotg_cl_offload ( a, b, c, s ) BIND(C)
       double precision,intent(inout)      :: a( * ), b( * ), c( * ), s( * )
     end subroutine mkl_blas_drotg_cl_offload

     subroutine mkl_blas_srotg_cl_offload ( a, b, c, s ) BIND(C)
       real,intent(inout)                  :: a( * ), b( * ), c( * ), s( * )
     end subroutine mkl_blas_srotg_cl_offload

     subroutine mkl_blas_zrotg_cl_offload ( a, b, c, s ) BIND(C)
       complex*16,intent(inout)            :: a( * ), b( * ), s( * )
       double precision,intent(inout)      :: c( * )
     end subroutine mkl_blas_zrotg_cl_offload

     subroutine mkl_blas_crotg_cl_offload ( a, b, c, s ) BIND(C)
       complex,intent(inout)               :: a( * ), b( * ), s( * )
       real,intent(inout)                  :: c( * )
     end subroutine mkl_blas_crotg_cl_offload

     subroutine mkl_blas_drotm_cl_offload ( n, x, incx,   &
          &y, incy, param ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(in)         :: param( * )
       double precision,intent(inout)      :: x( * ), y( * )
     end subroutine mkl_blas_drotm_cl_offload

     subroutine mkl_blas_srotm_cl_offload ( n, x, incx,   &
          &y, incy, param ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       real,intent(in)                     :: param( * )
       real,intent(inout)                  :: x( * ), y( * )
     end subroutine mkl_blas_srotm_cl_offload

     subroutine mkl_blas_drotmg_cl_offload ( d1, d2, x1, x2, param ) BIND(C)
       double precision,intent(in)         :: x2
       double precision,intent(inout)      :: d1( * ), d2( * ), x1( * ), param( * )
     end subroutine mkl_blas_drotmg_cl_offload

     subroutine mkl_blas_srotmg_cl_offload ( d1, d2, x1, x2, param ) BIND(C)
       real,intent(in)                     :: x2
       real,intent(inout)                  :: d1( * ), d2( * ), x1( * ), param( * )
     end subroutine mkl_blas_srotmg_cl_offload

     subroutine mkl_blas_dscal_cl_offload ( n, alpha, x, incx ) BIND(C)
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, n
       double precision,intent(inout)      :: x( * )
     end subroutine mkl_blas_dscal_cl_offload

     subroutine mkl_blas_sscal_cl_offload ( n, alpha, x, incx ) BIND(C)
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, n
       real,intent(inout)                  :: x( * )
     end subroutine mkl_blas_sscal_cl_offload

     subroutine mkl_blas_zscal_cl_offload ( n, alpha, x, incx ) BIND(C)
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, n
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_zscal_cl_offload

     subroutine mkl_blas_cscal_cl_offload ( n, alpha, x, incx ) BIND(C)
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, n
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_cscal_cl_offload

     subroutine mkl_blas_zdscal_cl_offload ( n, alpha, x, incx ) BIND(C)
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, n
       complex*16,intent(inout)            :: x( * )
     end subroutine mkl_blas_zdscal_cl_offload

     subroutine mkl_blas_csscal_cl_offload ( n, alpha, x, incx ) BIND(C)
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, n
       complex,intent(inout)               :: x( * )
     end subroutine mkl_blas_csscal_cl_offload

     subroutine mkl_blas_dswap_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       double precision,intent(inout)      :: x( * ), y( * )
     end subroutine mkl_blas_dswap_cl_offload

     subroutine mkl_blas_sswap_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       real,intent(inout)                  :: x( * ), y( * )
     end subroutine mkl_blas_sswap_cl_offload

     subroutine mkl_blas_zswap_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       complex*16,intent(inout)            :: x( * ), y( * )
     end subroutine mkl_blas_zswap_cl_offload

     subroutine mkl_blas_cswap_cl_offload ( n, x, incx,   &
          &y, incy ) BIND(C)
       integer,intent(in)                  :: incx, incy, n
       complex,intent(inout)               :: x( * ), y( * )
     end subroutine mkl_blas_cswap_cl_offload

     function mkl_blas_idamax_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: idamax
       integer,intent(in)                  :: incx, n
       double precision,intent(in)         :: x( * )
     end function mkl_blas_idamax_cl_offload

     function mkl_blas_isamax_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: isamax
       integer,intent(in)                  :: incx, n
       real,intent(in)                     :: x( * )
     end function mkl_blas_isamax_cl_offload

     function mkl_blas_izamax_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: izamax
       integer,intent(in)                  :: incx, n
       complex*16,intent(in)               :: x( * )
     end function mkl_blas_izamax_cl_offload

     function mkl_blas_icamax_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: icamax
       integer,intent(in)                  :: incx, n
       complex,intent(in)                  :: x( * )
     end function mkl_blas_icamax_cl_offload

     function mkl_blas_idamin_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: idamin
       integer,intent(in)                  :: incx, n
       double precision,intent(in)         :: x( * )
     end function mkl_blas_idamin_cl_offload

     function mkl_blas_isamin_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: isamin
       integer,intent(in)                  :: incx, n
       real,intent(in)                     :: x( * )
     end function mkl_blas_isamin_cl_offload

     function mkl_blas_izamin_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: izamin
       integer,intent(in)                  :: incx, n
       complex*16,intent(in)               :: x( * )
     end function mkl_blas_izamin_cl_offload

     function mkl_blas_icamin_cl_offload ( n, x, incx ) BIND(C)
       integer                             :: icamin
       integer,intent(in)                  :: incx, n
       complex,intent(in)                  :: x( * )
     end function mkl_blas_icamin_cl_offload

     ! BLAS batch

     subroutine mkl_blas_dtrsm_batch_strided_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda, stridea,        &
          &b, ldb, strideb, batch_size ) BIND(C)
       character*1,intent(in)             :: side, uplo, trans, diag
       integer,intent(in)                 :: m, n, lda, stridea, ldb, strideb, batch_size
       double precision,intent(in)        :: alpha
       double precision,intent(in)        :: a( lda, * )
       double precision,intent(inout)     :: b( ldb, * )
     end subroutine mkl_blas_dtrsm_batch_strided_cl_offload

     subroutine mkl_blas_strsm_batch_strided_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda, stridea,        &
          &b, ldb, strideb, batch_size ) BIND(C)
       character*1,intent(in)              :: side, uplo, trans, diag
       integer,intent(in)                  :: m, n, lda, stridea, ldb, strideb, batch_size
       real,intent(in)                     :: alpha
       real,intent(in)                     :: a( lda, * )
       real,intent(in)                     :: b( ldb, * )
     end subroutine mkl_blas_strsm_batch_strided_cl_offload

     subroutine mkl_blas_ztrsm_batch_strided_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda, stridea,        &
          &b, ldb, strideb, batch_size ) BIND(C)
       character*1,intent(in)             :: side, uplo, trans, diag
       integer,intent(in)                 :: m, n, lda, stridea, ldb, strideb, batch_size
       complex*16,intent(in)              :: alpha
       complex*16,intent(in)              :: a( lda, * )
       complex*16,intent(inout)           :: b( ldb, * )
     end subroutine mkl_blas_ztrsm_batch_strided_cl_offload

     subroutine mkl_blas_ctrsm_batch_strided_cl_offload ( side, uplo, trans, diag, m, n, alpha, a, lda, stridea,        &
          &b, ldb, strideb, batch_size ) BIND(C)
       character*1,intent(in)              :: side, uplo, trans, diag
       integer,intent(in)                  :: m, n, lda, stridea, ldb, strideb, batch_size
       complex,intent(in)                  :: alpha
       complex,intent(in)                  :: a( lda, * )
       complex,intent(in)                  :: b( ldb, * )
     end subroutine mkl_blas_ctrsm_batch_strided_cl_offload

     subroutine mkl_blas_dgemm_batch_strided_cl_offload ( transa, transb, m, n, k, alpha, a, lda, stridea,        &
          &b, ldb, strideb, beta, c, ldc, stridec, batch_size ) BIND(C)
       character*1,intent(in)             :: transa, transb
       integer,intent(in)                 :: m, n, k, lda, stridea, ldb, strideb, ldc, stridec, batch_size
       double precision,intent(in)        :: alpha, beta
       double precision,intent(in)        :: a( lda, * ), b( ldb, * )
       double precision,intent(inout)     :: c( ldc, * )
     end subroutine mkl_blas_dgemm_batch_strided_cl_offload

     subroutine mkl_blas_sgemm_batch_strided_cl_offload ( transa, transb, m, n, k, alpha, a, lda, stridea,        &
          &b, ldb, strideb, beta, c, ldc, stridec, batch_size ) BIND(C)
       character*1,intent(in)              :: transa, transb
       integer,intent(in)                  :: m, n, k, lda, stridea, ldb, strideb, ldc, stridec, batch_size
       real,intent(in)                     :: alpha, beta
       real,intent(in)                     :: a( lda, * ), b( ldb, * )
       real,intent(in)                     :: c( ldc, * )
     end subroutine mkl_blas_sgemm_batch_strided_cl_offload

     subroutine mkl_blas_zgemm_batch_strided_cl_offload ( transa, transb, m, n, k, alpha, a, lda, stridea,        &
          &b, ldb, strideb, beta, c, ldc, stridec, batch_size ) BIND(C)
       character*1,intent(in)             :: transa, transb
       integer,intent(in)                 :: m, n, k, lda, stridea, ldb, strideb, ldc, stridec, batch_size
       complex*16,intent(in)              :: alpha, beta
       complex*16,intent(in)              :: a( lda, * ), b( ldb, * )
       complex*16,intent(inout)           :: c( ldc, * )
     end subroutine mkl_blas_zgemm_batch_strided_cl_offload

     subroutine mkl_blas_cgemm_batch_strided_cl_offload ( transa, transb, m, n, k, alpha, a, lda, stridea,        &
          &b, ldb, strideb, beta, c, ldc, stridec, batch_size ) BIND(C)
       character*1,intent(in)              :: transa, transb
       integer,intent(in)                  :: m, n, k, lda, stridea, ldb, strideb, ldc, stridec, batch_size
       complex,intent(in)                  :: alpha, beta
       complex,intent(in)                  :: a( lda, * ), b( ldb, * )
       complex,intent(in)                  :: c( ldc, * )
     end subroutine mkl_blas_cgemm_batch_strided_cl_offload

    subroutine mkl_blas_daxpy_batch_strided_cl_offload ( n, alpha, x, incx, stridex,   &
          &y, incy, stridey, batch_size ) BIND(C)
       double precision,intent(in)         :: alpha
       integer,intent(in)                  :: incx, stridex, incy, stridey, batch_size, n
       double precision,intent(in)         :: x( * )
       double precision,intent(inout)      :: y( * )
     end subroutine mkl_blas_daxpy_batch_strided_cl_offload

     subroutine mkl_blas_saxpy_batch_strided_cl_offload ( n, alpha, x, incx, stridex,   &
          &y, incy, stridey, batch_size ) BIND(C)
       real,intent(in)                     :: alpha
       integer,intent(in)                  :: incx, stridex, incy, stridey, batch_size, n
       real,intent(in)                     :: x( * )
       real,intent(inout)                  :: y( * )
     end subroutine mkl_blas_saxpy_batch_strided_cl_offload

     subroutine mkl_blas_zaxpy_batch_strided_cl_offload ( n, alpha, x, incx, stridex,   &
          &y, incy, stridey, batch_size ) BIND(C)
       complex*16,intent(in)               :: alpha
       integer,intent(in)                  :: incx, stridex, incy, stridey, batch_size, n
       complex*16,intent(in)               :: x( * )
       complex*16,intent(inout)            :: y( * )
     end subroutine mkl_blas_zaxpy_batch_strided_cl_offload

     subroutine mkl_blas_caxpy_batch_strided_cl_offload ( n, alpha, x, incx, stridex,   &
          &y, incy, stridey, batch_size ) BIND(C)
       complex,intent(in)                  :: alpha
       integer,intent(in)                  :: incx, stridex, incy, stridey, batch_size, n
       complex,intent(in)                  :: x( * )
       complex,intent(inout)               :: y( * )
     end subroutine mkl_blas_caxpy_batch_strided_cl_offload

  end interface
