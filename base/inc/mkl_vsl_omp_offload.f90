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
!      Intel(R) oneMKL Library FORTRAN interface for VSL OpenMP offload
!*******************************************************************************

include "mkl_vsl_omp_variant.f90"

module onemkl_vsl_omp_offload
  use onemkl_vsl_omp_variant

  !++
  !  VSL CONTINUOUS DISTRIBUTION GENERATOR FUNCTION INTERFACES.
  !--

  ! Uniform distribution
  interface
     integer function vsrnguniform(method, stream, n, r, a, b)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: b
      !$omp  declare variant( vsrnguniform:mkl_vsl_vsrnguniform_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrnguniform
  end interface

  interface
     integer function vdrnguniform (method, stream, n, r, a, b)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: b
       !$omp  declare variant( vdrnguniform:mkl_vsl_vdrnguniform_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrnguniform
  end interface
  
  ! Gaussian distribution
  interface
     integer function vsrnggaussian (method, stream, n, r, a, sigma)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: sigma
       !$omp  declare variant( vsrnggaussian:mkl_vsl_vsrnggaussian_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrnggaussian
  end interface

  interface
     integer function vdrnggaussian (method, stream, n, r, a, sigma)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: sigma
       !$omp  declare variant( vdrnggaussian:mkl_vsl_vdrnggaussian_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrnggaussian
  end interface

  ! Exponential distribution
  interface
     integer function vsrngexponential (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrngexponential:mkl_vsl_vsrngexponential_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrngexponential
  end interface

  interface
     integer function vdrngexponential (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrngexponential:mkl_vsl_vdrngexponential_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrngexponential
  end interface

  ! Laplace distribution
  interface
     integer function vsrnglaplace (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrnglaplace:mkl_vsl_vsrnglaplace_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrnglaplace
  end interface

  interface
     integer function vdrnglaplace (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrnglaplace:mkl_vsl_vdrnglaplace_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrnglaplace
  end interface

  ! Weibull distribution
  interface
     integer function vsrngweibull (method, stream, n, r, alpha, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: alpha
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrngweibull:mkl_vsl_vsrngweibull_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrngweibull
  end interface

  interface
     integer function vdrngweibull (method, stream, n, r, alpha, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: alpha
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrngweibull:mkl_vsl_vdrngweibull_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrngweibull
  end interface

  ! Cauchy distribution
  interface
     integer function vsrngcauchy (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrngcauchy:mkl_vsl_vsrngcauchy_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrngcauchy
  end interface

  interface
     integer function vdrngcauchy (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrngcauchy:mkl_vsl_vdrngcauchy_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrngcauchy
  end interface

  ! Rayleigh distribution
  interface
     integer function vsrngrayleigh (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrngrayleigh:mkl_vsl_vsrngrayleigh_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrngrayleigh
  end interface

  interface
     integer function vdrngrayleigh (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrngrayleigh:mkl_vsl_vdrngrayleigh_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrngrayleigh
  end interface

  ! Lognormal distribution
  interface
     integer function vsrnglognormal (method, stream, n, r, a, sigma, b, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: sigma
       real(kind=4),intent(in)  :: b
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrnglognormal:mkl_vsl_vsrnglognormal_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrnglognormal
  end interface

  interface
     integer function vdrnglognormal (method, stream, n, r, a, sigma, b, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: sigma
       real(kind=8),intent(in)  :: b
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrnglognormal:mkl_vsl_vdrnglognormal_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrnglognormal
  end interface

  ! Gumbel distribution
  interface
     integer function vsrnggumbel (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=4),intent(out) :: r(n)
       real(kind=4),intent(in)  :: a
       real(kind=4),intent(in)  :: beta
       !$omp  declare variant( vsrnggumbel:mkl_vsl_vsrnggumbel_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vsrnggumbel
  end interface

  interface
     integer function vdrnggumbel (method, stream, n, r, a, beta)
       import :: VSL_STREAM_STATE
       integer,intent(in)       :: method
       type(VSL_STREAM_STATE)   :: stream
       integer,intent(in)       :: n
       real(kind=8),intent(out) :: r(n)
       real(kind=8),intent(in)  :: a
       real(kind=8),intent(in)  :: beta
       !$omp  declare variant( vdrnggumbel:mkl_vsl_vdrnggumbel_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function vdrnggumbel
  end interface

!++
!  VSL DISCRETE DISTRIBUTION GENERATOR FUNCTION INTERFACES.
!--
  ! Uniform distribution
  interface
     integer function virnguniform (method, stream, n, r, a, b)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=4),intent(out) :: r(n)
       integer(kind=4),intent(in)  :: a
       integer(kind=4),intent(in)  :: b
       !$omp  declare variant( virnguniform:mkl_vsl_virnguniform_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virnguniform
  end interface

  ! UniformBits distribution
  interface
     integer function virnguniformbits (method, stream, n, r)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=4),intent(out) :: r(n)
       !$omp  declare variant( virnguniformbits:mkl_vsl_virnguniformbits_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virnguniformbits
  end interface

  ! UniformBits32 distribution
  interface
     integer function virnguniformbits32 (method, stream, n, r)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=4),intent(out) :: r(n)
       !$omp  declare variant( virnguniformbits32:mkl_vsl_virnguniformbits32_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virnguniformbits32
  end interface

  ! UniformBits64 distribution
  interface
     integer function virnguniformbits64 (method, stream, n, r)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=8),intent(out) :: r(n)
       !$omp  declare variant( virnguniformbits64:mkl_vsl_virnguniformbits64_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virnguniformbits64
  end interface

  ! Bernoulli distribution
  interface
     integer function virngbernoulli (method, stream, n, r, p)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=4),intent(out) :: r(n)
       real(kind=8),intent(in)     :: p
       !$omp  declare variant( virngbernoulli:mkl_vsl_virngbernoulli_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virngbernoulli
  end interface

  ! Geometric distribution
  interface
     integer function virnggeometric (method, stream, n, r, p)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=4),intent(out) :: r(n)
       real(kind=8),intent(in)     :: p
       !$omp  declare variant( virnggeometric:mkl_vsl_virnggeometric_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virnggeometric
  end interface

  ! Poisson distribution
  interface
     integer function virngpoisson (method, stream, n, r, lambda)
       import :: VSL_STREAM_STATE
       integer,intent(in)          :: method
       type(VSL_STREAM_STATE)      :: stream
       integer,intent(in)          :: n
       integer(kind=4),intent(out) :: r(n)
       real(kind=8),intent(in)     :: lambda
       !$omp  declare variant( virngpoisson:mkl_vsl_virngpoisson_omp_offload ) match( construct={target variant dispatch}, device={arch(gen)} )
     end function virngpoisson
  end interface

end module onemkl_vsl_omp_offload
