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
!      OpenMP offload for VM
!*******************************************************************************



module onemkl_vml_omp_offload


        integer(kind=8) ::vml_la
        integer(kind=8) ::vml_ha
        integer(kind=8) ::vml_ep
        parameter (vml_la = Z"00000001")
        parameter (vml_ha = Z"00000002")
        parameter (vml_ep = Z"00000003")



    include "mkl_vml_omp_variant.f90"

    interface
        integer(kind=8) function vmlsetmode(n)
        integer(kind=8), intent(in) :: n
!$omp declare variant ( vmlsetmode:mkl_vm_vmlsetmode_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end function

        subroutine vcabs(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcabs:mkl_vm_vcabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcabs

        subroutine vmcabs(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcabs:mkl_vm_vmcabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcabs

        subroutine vzabs(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzabs:mkl_vm_vzabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzabs

        subroutine vmzabs(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzabs:mkl_vm_vmzabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzabs

        subroutine vsabs(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsabs:mkl_vm_vsabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsabs

        subroutine vmsabs(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsabs:mkl_vm_vmsabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsabs

        subroutine vdabs(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdabs:mkl_vm_vdabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdabs

        subroutine vmdabs(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdabs:mkl_vm_vmdabs_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdabs


        subroutine vcacos(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcacos:mkl_vm_vcacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcacos

        subroutine vmcacos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcacos:mkl_vm_vmcacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcacos

        subroutine vzacos(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzacos:mkl_vm_vzacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzacos

        subroutine vmzacos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzacos:mkl_vm_vmzacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzacos

        subroutine vsacos(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsacos:mkl_vm_vsacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsacos

        subroutine vmsacos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsacos:mkl_vm_vmsacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsacos

        subroutine vdacos(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdacos:mkl_vm_vdacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdacos

        subroutine vmdacos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdacos:mkl_vm_vmdacos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdacos


        subroutine vcacosh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcacosh:mkl_vm_vcacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcacosh

        subroutine vmcacosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcacosh:mkl_vm_vmcacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcacosh

        subroutine vzacosh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzacosh:mkl_vm_vzacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzacosh

        subroutine vmzacosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzacosh:mkl_vm_vmzacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzacosh

        subroutine vsacosh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsacosh:mkl_vm_vsacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsacosh

        subroutine vmsacosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsacosh:mkl_vm_vmsacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsacosh

        subroutine vdacosh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdacosh:mkl_vm_vdacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdacosh

        subroutine vmdacosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdacosh:mkl_vm_vmdacosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdacosh


        subroutine vsacospi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsacospi:mkl_vm_vsacospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsacospi

        subroutine vmsacospi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsacospi:mkl_vm_vmsacospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsacospi

        subroutine vdacospi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdacospi:mkl_vm_vdacospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdacospi

        subroutine vmdacospi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdacospi:mkl_vm_vmdacospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdacospi


        subroutine vcadd(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcadd:mkl_vm_vcadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcadd

        subroutine vmcadd(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcadd:mkl_vm_vmcadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcadd

        subroutine vzadd(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzadd:mkl_vm_vzadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzadd

        subroutine vmzadd(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzadd:mkl_vm_vmzadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzadd

        subroutine vsadd(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsadd:mkl_vm_vsadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsadd

        subroutine vmsadd(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsadd:mkl_vm_vmsadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsadd

        subroutine vdadd(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdadd:mkl_vm_vdadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdadd

        subroutine vmdadd(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdadd:mkl_vm_vmdadd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdadd


        subroutine vcarg(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcarg:mkl_vm_vcarg_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcarg

        subroutine vmcarg(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcarg:mkl_vm_vmcarg_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcarg

        subroutine vzarg(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzarg:mkl_vm_vzarg_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzarg

        subroutine vmzarg(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzarg:mkl_vm_vmzarg_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzarg


        subroutine vcasin(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcasin:mkl_vm_vcasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcasin

        subroutine vmcasin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcasin:mkl_vm_vmcasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcasin

        subroutine vzasin(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzasin:mkl_vm_vzasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzasin

        subroutine vmzasin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzasin:mkl_vm_vmzasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzasin

        subroutine vsasin(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsasin:mkl_vm_vsasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsasin

        subroutine vmsasin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsasin:mkl_vm_vmsasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsasin

        subroutine vdasin(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdasin:mkl_vm_vdasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdasin

        subroutine vmdasin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdasin:mkl_vm_vmdasin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdasin


        subroutine vcasinh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcasinh:mkl_vm_vcasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcasinh

        subroutine vmcasinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcasinh:mkl_vm_vmcasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcasinh

        subroutine vzasinh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzasinh:mkl_vm_vzasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzasinh

        subroutine vmzasinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzasinh:mkl_vm_vmzasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzasinh

        subroutine vsasinh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsasinh:mkl_vm_vsasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsasinh

        subroutine vmsasinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsasinh:mkl_vm_vmsasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsasinh

        subroutine vdasinh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdasinh:mkl_vm_vdasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdasinh

        subroutine vmdasinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdasinh:mkl_vm_vmdasinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdasinh


        subroutine vsasinpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsasinpi:mkl_vm_vsasinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsasinpi

        subroutine vmsasinpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsasinpi:mkl_vm_vmsasinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsasinpi

        subroutine vdasinpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdasinpi:mkl_vm_vdasinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdasinpi

        subroutine vmdasinpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdasinpi:mkl_vm_vmdasinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdasinpi


        subroutine vcatan(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcatan:mkl_vm_vcatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcatan

        subroutine vmcatan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcatan:mkl_vm_vmcatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcatan

        subroutine vzatan(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzatan:mkl_vm_vzatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzatan

        subroutine vmzatan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzatan:mkl_vm_vmzatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzatan

        subroutine vsatan(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsatan:mkl_vm_vsatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsatan

        subroutine vmsatan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsatan:mkl_vm_vmsatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsatan

        subroutine vdatan(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdatan:mkl_vm_vdatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdatan

        subroutine vmdatan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdatan:mkl_vm_vmdatan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdatan


        subroutine vsatan2(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsatan2:mkl_vm_vsatan2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsatan2

        subroutine vmsatan2(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsatan2:mkl_vm_vmsatan2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsatan2

        subroutine vdatan2(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdatan2:mkl_vm_vdatan2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdatan2

        subroutine vmdatan2(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdatan2:mkl_vm_vmdatan2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdatan2


        subroutine vsatan2pi(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsatan2pi:mkl_vm_vsatan2pi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsatan2pi

        subroutine vmsatan2pi(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsatan2pi:mkl_vm_vmsatan2pi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsatan2pi

        subroutine vdatan2pi(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdatan2pi:mkl_vm_vdatan2pi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdatan2pi

        subroutine vmdatan2pi(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdatan2pi:mkl_vm_vmdatan2pi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdatan2pi


        subroutine vcatanh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcatanh:mkl_vm_vcatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcatanh

        subroutine vmcatanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcatanh:mkl_vm_vmcatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcatanh

        subroutine vzatanh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzatanh:mkl_vm_vzatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzatanh

        subroutine vmzatanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzatanh:mkl_vm_vmzatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzatanh

        subroutine vsatanh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsatanh:mkl_vm_vsatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsatanh

        subroutine vmsatanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsatanh:mkl_vm_vmsatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsatanh

        subroutine vdatanh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdatanh:mkl_vm_vdatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdatanh

        subroutine vmdatanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdatanh:mkl_vm_vmdatanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdatanh


        subroutine vsatanpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsatanpi:mkl_vm_vsatanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsatanpi

        subroutine vmsatanpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsatanpi:mkl_vm_vmsatanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsatanpi

        subroutine vdatanpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdatanpi:mkl_vm_vdatanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdatanpi

        subroutine vmdatanpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdatanpi:mkl_vm_vmdatanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdatanpi


        subroutine vscbrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscbrt:mkl_vm_vscbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscbrt

        subroutine vmscbrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscbrt:mkl_vm_vmscbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscbrt

        subroutine vdcbrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcbrt:mkl_vm_vdcbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcbrt

        subroutine vmdcbrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcbrt:mkl_vm_vmdcbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcbrt


        subroutine vscdfnorm(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscdfnorm:mkl_vm_vscdfnorm_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscdfnorm

        subroutine vmscdfnorm(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscdfnorm:mkl_vm_vmscdfnorm_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscdfnorm

        subroutine vdcdfnorm(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcdfnorm:mkl_vm_vdcdfnorm_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcdfnorm

        subroutine vmdcdfnorm(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcdfnorm:mkl_vm_vmdcdfnorm_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcdfnorm


        subroutine vscdfnorminv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscdfnorminv:mkl_vm_vscdfnorminv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscdfnorminv

        subroutine vmscdfnorminv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscdfnorminv:mkl_vm_vmscdfnorminv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscdfnorminv

        subroutine vdcdfnorminv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcdfnorminv:mkl_vm_vdcdfnorminv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcdfnorminv

        subroutine vmdcdfnorminv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcdfnorminv:mkl_vm_vmdcdfnorminv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcdfnorminv


        subroutine vsceil(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsceil:mkl_vm_vsceil_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsceil

        subroutine vmsceil(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsceil:mkl_vm_vmsceil_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsceil

        subroutine vdceil(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdceil:mkl_vm_vdceil_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdceil

        subroutine vmdceil(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdceil:mkl_vm_vmdceil_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdceil


        subroutine vccis(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vccis:mkl_vm_vccis_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vccis

        subroutine vmccis(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmccis:mkl_vm_vmccis_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmccis

        subroutine vzcis(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzcis:mkl_vm_vzcis_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzcis

        subroutine vmzcis(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzcis:mkl_vm_vmzcis_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzcis


        subroutine vcconj(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcconj:mkl_vm_vcconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcconj

        subroutine vmcconj(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcconj:mkl_vm_vmcconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcconj

        subroutine vzconj(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzconj:mkl_vm_vzconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzconj

        subroutine vmzconj(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzconj:mkl_vm_vmzconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzconj


        subroutine vscopysign(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscopysign:mkl_vm_vscopysign_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscopysign

        subroutine vmscopysign(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscopysign:mkl_vm_vmscopysign_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscopysign

        subroutine vdcopysign(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcopysign:mkl_vm_vdcopysign_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcopysign

        subroutine vmdcopysign(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcopysign:mkl_vm_vmdcopysign_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcopysign


        subroutine vccos(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vccos:mkl_vm_vccos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vccos

        subroutine vmccos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmccos:mkl_vm_vmccos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmccos

        subroutine vzcos(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzcos:mkl_vm_vzcos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzcos

        subroutine vmzcos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzcos:mkl_vm_vmzcos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzcos

        subroutine vscos(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscos:mkl_vm_vscos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscos

        subroutine vmscos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscos:mkl_vm_vmscos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscos

        subroutine vdcos(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcos:mkl_vm_vdcos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcos

        subroutine vmdcos(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcos:mkl_vm_vmdcos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcos


        subroutine vscosd(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscosd:mkl_vm_vscosd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscosd

        subroutine vmscosd(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscosd:mkl_vm_vmscosd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscosd

        subroutine vdcosd(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcosd:mkl_vm_vdcosd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcosd

        subroutine vmdcosd(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcosd:mkl_vm_vmdcosd_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcosd


        subroutine vccosh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vccosh:mkl_vm_vccosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vccosh

        subroutine vmccosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmccosh:mkl_vm_vmccosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmccosh

        subroutine vzcosh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzcosh:mkl_vm_vzcosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzcosh

        subroutine vmzcosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzcosh:mkl_vm_vmzcosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzcosh

        subroutine vscosh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscosh:mkl_vm_vscosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscosh

        subroutine vmscosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscosh:mkl_vm_vmscosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscosh

        subroutine vdcosh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcosh:mkl_vm_vdcosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcosh

        subroutine vmdcosh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcosh:mkl_vm_vmdcosh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcosh


        subroutine vscospi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vscospi:mkl_vm_vscospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vscospi

        subroutine vmscospi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmscospi:mkl_vm_vmscospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmscospi

        subroutine vdcospi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdcospi:mkl_vm_vdcospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdcospi

        subroutine vmdcospi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdcospi:mkl_vm_vmdcospi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdcospi


        subroutine vcdiv(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcdiv:mkl_vm_vcdiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcdiv

        subroutine vmcdiv(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcdiv:mkl_vm_vmcdiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcdiv

        subroutine vzdiv(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzdiv:mkl_vm_vzdiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzdiv

        subroutine vmzdiv(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzdiv:mkl_vm_vmzdiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzdiv

        subroutine vsdiv(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsdiv:mkl_vm_vsdiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsdiv

        subroutine vmsdiv(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsdiv:mkl_vm_vmsdiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsdiv

        subroutine vddiv(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vddiv:mkl_vm_vddiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vddiv

        subroutine vmddiv(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmddiv:mkl_vm_vmddiv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmddiv


        subroutine vserf(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vserf:mkl_vm_vserf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vserf

        subroutine vmserf(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmserf:mkl_vm_vmserf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmserf

        subroutine vderf(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vderf:mkl_vm_vderf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vderf

        subroutine vmderf(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmderf:mkl_vm_vmderf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmderf


        subroutine vserfc(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vserfc:mkl_vm_vserfc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vserfc

        subroutine vmserfc(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmserfc:mkl_vm_vmserfc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmserfc

        subroutine vderfc(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vderfc:mkl_vm_vderfc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vderfc

        subroutine vmderfc(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmderfc:mkl_vm_vmderfc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmderfc


        subroutine vserfcinv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vserfcinv:mkl_vm_vserfcinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vserfcinv

        subroutine vmserfcinv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmserfcinv:mkl_vm_vmserfcinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmserfcinv

        subroutine vderfcinv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vderfcinv:mkl_vm_vderfcinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vderfcinv

        subroutine vmderfcinv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmderfcinv:mkl_vm_vmderfcinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmderfcinv


        subroutine vserfinv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vserfinv:mkl_vm_vserfinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vserfinv

        subroutine vmserfinv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmserfinv:mkl_vm_vmserfinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmserfinv

        subroutine vderfinv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vderfinv:mkl_vm_vderfinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vderfinv

        subroutine vmderfinv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmderfinv:mkl_vm_vmderfinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmderfinv


        subroutine vcexp(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcexp:mkl_vm_vcexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcexp

        subroutine vmcexp(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcexp:mkl_vm_vmcexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcexp

        subroutine vzexp(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzexp:mkl_vm_vzexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzexp

        subroutine vmzexp(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzexp:mkl_vm_vmzexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzexp

        subroutine vsexp(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsexp:mkl_vm_vsexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsexp

        subroutine vmsexp(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsexp:mkl_vm_vmsexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsexp

        subroutine vdexp(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdexp:mkl_vm_vdexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdexp

        subroutine vmdexp(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdexp:mkl_vm_vmdexp_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdexp


        subroutine vsexp2(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsexp2:mkl_vm_vsexp2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsexp2

        subroutine vmsexp2(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsexp2:mkl_vm_vmsexp2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsexp2

        subroutine vdexp2(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdexp2:mkl_vm_vdexp2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdexp2

        subroutine vmdexp2(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdexp2:mkl_vm_vmdexp2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdexp2


        subroutine vsexp10(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsexp10:mkl_vm_vsexp10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsexp10

        subroutine vmsexp10(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsexp10:mkl_vm_vmsexp10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsexp10

        subroutine vdexp10(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdexp10:mkl_vm_vdexp10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdexp10

        subroutine vmdexp10(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdexp10:mkl_vm_vmdexp10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdexp10


        subroutine vsexpint1(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsexpint1:mkl_vm_vsexpint1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsexpint1

        subroutine vmsexpint1(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsexpint1:mkl_vm_vmsexpint1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsexpint1

        subroutine vdexpint1(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdexpint1:mkl_vm_vdexpint1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdexpint1

        subroutine vmdexpint1(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdexpint1:mkl_vm_vmdexpint1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdexpint1


        subroutine vsexpm1(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsexpm1:mkl_vm_vsexpm1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsexpm1

        subroutine vmsexpm1(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsexpm1:mkl_vm_vmsexpm1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsexpm1

        subroutine vdexpm1(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdexpm1:mkl_vm_vdexpm1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdexpm1

        subroutine vmdexpm1(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdexpm1:mkl_vm_vmdexpm1_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdexpm1


        subroutine vsfdim(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsfdim:mkl_vm_vsfdim_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsfdim

        subroutine vmsfdim(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsfdim:mkl_vm_vmsfdim_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsfdim

        subroutine vdfdim(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdfdim:mkl_vm_vdfdim_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdfdim

        subroutine vmdfdim(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdfdim:mkl_vm_vmdfdim_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdfdim


        subroutine vsfloor(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsfloor:mkl_vm_vsfloor_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsfloor

        subroutine vmsfloor(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsfloor:mkl_vm_vmsfloor_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsfloor

        subroutine vdfloor(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdfloor:mkl_vm_vdfloor_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdfloor

        subroutine vmdfloor(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdfloor:mkl_vm_vmdfloor_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdfloor


        subroutine vsfmax(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsfmax:mkl_vm_vsfmax_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsfmax

        subroutine vmsfmax(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsfmax:mkl_vm_vmsfmax_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsfmax

        subroutine vdfmax(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdfmax:mkl_vm_vdfmax_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdfmax

        subroutine vmdfmax(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdfmax:mkl_vm_vmdfmax_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdfmax


        subroutine vsfmin(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsfmin:mkl_vm_vsfmin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsfmin

        subroutine vmsfmin(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsfmin:mkl_vm_vmsfmin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsfmin

        subroutine vdfmin(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdfmin:mkl_vm_vdfmin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdfmin

        subroutine vmdfmin(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdfmin:mkl_vm_vmdfmin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdfmin


        subroutine vsfmod(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsfmod:mkl_vm_vsfmod_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsfmod

        subroutine vmsfmod(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsfmod:mkl_vm_vmsfmod_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsfmod

        subroutine vdfmod(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdfmod:mkl_vm_vdfmod_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdfmod

        subroutine vmdfmod(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdfmod:mkl_vm_vmdfmod_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdfmod


        subroutine vsfrac(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsfrac:mkl_vm_vsfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsfrac

        subroutine vmsfrac(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsfrac:mkl_vm_vmsfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsfrac

        subroutine vdfrac(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdfrac:mkl_vm_vdfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdfrac

        subroutine vmdfrac(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdfrac:mkl_vm_vmdfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdfrac


        subroutine vshypot(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vshypot:mkl_vm_vshypot_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vshypot

        subroutine vmshypot(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmshypot:mkl_vm_vmshypot_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmshypot

        subroutine vdhypot(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdhypot:mkl_vm_vdhypot_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdhypot

        subroutine vmdhypot(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdhypot:mkl_vm_vmdhypot_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdhypot


        subroutine vsinv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsinv:mkl_vm_vsinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsinv

        subroutine vmsinv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsinv:mkl_vm_vmsinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsinv

        subroutine vdinv(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdinv:mkl_vm_vdinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdinv

        subroutine vmdinv(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdinv:mkl_vm_vmdinv_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdinv


        subroutine vsinvcbrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsinvcbrt:mkl_vm_vsinvcbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsinvcbrt

        subroutine vmsinvcbrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsinvcbrt:mkl_vm_vmsinvcbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsinvcbrt

        subroutine vdinvcbrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdinvcbrt:mkl_vm_vdinvcbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdinvcbrt

        subroutine vmdinvcbrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdinvcbrt:mkl_vm_vmdinvcbrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdinvcbrt


        subroutine vsinvsqrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsinvsqrt:mkl_vm_vsinvsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsinvsqrt

        subroutine vmsinvsqrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsinvsqrt:mkl_vm_vmsinvsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsinvsqrt

        subroutine vdinvsqrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdinvsqrt:mkl_vm_vdinvsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdinvsqrt

        subroutine vmdinvsqrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdinvsqrt:mkl_vm_vmdinvsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdinvsqrt


        subroutine vslgamma(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vslgamma:mkl_vm_vslgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vslgamma

        subroutine vmslgamma(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmslgamma:mkl_vm_vmslgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmslgamma

        subroutine vdlgamma(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdlgamma:mkl_vm_vdlgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdlgamma

        subroutine vmdlgamma(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdlgamma:mkl_vm_vmdlgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdlgamma


        subroutine vslinearfrac(n, a, b, c, d, e, f, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: c
        real(kind=4), intent(inout) :: d
        real(kind=4), intent(inout) :: e
        real(kind=4), intent(inout) :: f
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vslinearfrac:mkl_vm_vslinearfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vslinearfrac

        subroutine vmslinearfrac(n, a, b, c, d, e, f, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: c
        real(kind=4), intent(inout) :: d
        real(kind=4), intent(inout) :: e
        real(kind=4), intent(inout) :: f
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmslinearfrac:mkl_vm_vmslinearfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmslinearfrac

        subroutine vdlinearfrac(n, a, b, c, d, e, f, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: c
        real(kind=8), intent(inout) :: d
        real(kind=8), intent(inout) :: e
        real(kind=8), intent(inout) :: f
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdlinearfrac:mkl_vm_vdlinearfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdlinearfrac

        subroutine vmdlinearfrac(n, a, b, c, d, e, f, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: c
        real(kind=8), intent(inout) :: d
        real(kind=8), intent(inout) :: e
        real(kind=8), intent(inout) :: f
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdlinearfrac:mkl_vm_vmdlinearfrac_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdlinearfrac


        subroutine vcln(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcln:mkl_vm_vcln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcln

        subroutine vmcln(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcln:mkl_vm_vmcln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcln

        subroutine vzln(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzln:mkl_vm_vzln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzln

        subroutine vmzln(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzln:mkl_vm_vmzln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzln

        subroutine vsln(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsln:mkl_vm_vsln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsln

        subroutine vmsln(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsln:mkl_vm_vmsln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsln

        subroutine vdln(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdln:mkl_vm_vdln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdln

        subroutine vmdln(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdln:mkl_vm_vmdln_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdln


        subroutine vslog2(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vslog2:mkl_vm_vslog2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vslog2

        subroutine vmslog2(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmslog2:mkl_vm_vmslog2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmslog2

        subroutine vdlog2(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdlog2:mkl_vm_vdlog2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdlog2

        subroutine vmdlog2(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdlog2:mkl_vm_vmdlog2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdlog2


        subroutine vslogb(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vslogb:mkl_vm_vslogb_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vslogb

        subroutine vmslogb(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmslogb:mkl_vm_vmslogb_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmslogb

        subroutine vdlogb(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdlogb:mkl_vm_vdlogb_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdlogb

        subroutine vmdlogb(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdlogb:mkl_vm_vmdlogb_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdlogb


        subroutine vclog10(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vclog10:mkl_vm_vclog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vclog10

        subroutine vmclog10(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmclog10:mkl_vm_vmclog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmclog10

        subroutine vzlog10(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzlog10:mkl_vm_vzlog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzlog10

        subroutine vmzlog10(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzlog10:mkl_vm_vmzlog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzlog10

        subroutine vslog10(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vslog10:mkl_vm_vslog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vslog10

        subroutine vmslog10(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmslog10:mkl_vm_vmslog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmslog10

        subroutine vdlog10(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdlog10:mkl_vm_vdlog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdlog10

        subroutine vmdlog10(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdlog10:mkl_vm_vmdlog10_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdlog10


        subroutine vslog1p(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vslog1p:mkl_vm_vslog1p_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vslog1p

        subroutine vmslog1p(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmslog1p:mkl_vm_vmslog1p_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmslog1p

        subroutine vdlog1p(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdlog1p:mkl_vm_vdlog1p_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdlog1p

        subroutine vmdlog1p(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdlog1p:mkl_vm_vmdlog1p_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdlog1p


        subroutine vsmaxmag(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsmaxmag:mkl_vm_vsmaxmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsmaxmag

        subroutine vmsmaxmag(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsmaxmag:mkl_vm_vmsmaxmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsmaxmag

        subroutine vdmaxmag(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdmaxmag:mkl_vm_vdmaxmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdmaxmag

        subroutine vmdmaxmag(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdmaxmag:mkl_vm_vmdmaxmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdmaxmag


        subroutine vsminmag(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsminmag:mkl_vm_vsminmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsminmag

        subroutine vmsminmag(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsminmag:mkl_vm_vmsminmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsminmag

        subroutine vdminmag(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdminmag:mkl_vm_vdminmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdminmag

        subroutine vmdminmag(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdminmag:mkl_vm_vmdminmag_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdminmag


        subroutine vsmodf(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
!$omp declare variant ( vsmodf:mkl_vm_vsmodf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsmodf

        subroutine vmsmodf(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsmodf:mkl_vm_vmsmodf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsmodf

        subroutine vdmodf(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
!$omp declare variant ( vdmodf:mkl_vm_vdmodf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdmodf

        subroutine vmdmodf(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdmodf:mkl_vm_vmdmodf_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdmodf


        subroutine vcmul(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcmul:mkl_vm_vcmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcmul

        subroutine vmcmul(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcmul:mkl_vm_vmcmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcmul

        subroutine vzmul(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzmul:mkl_vm_vzmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzmul

        subroutine vmzmul(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzmul:mkl_vm_vmzmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzmul

        subroutine vsmul(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsmul:mkl_vm_vsmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsmul

        subroutine vmsmul(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsmul:mkl_vm_vmsmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsmul

        subroutine vdmul(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdmul:mkl_vm_vdmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdmul

        subroutine vmdmul(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdmul:mkl_vm_vmdmul_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdmul


        subroutine vcmulbyconj(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcmulbyconj:mkl_vm_vcmulbyconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcmulbyconj

        subroutine vmcmulbyconj(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcmulbyconj:mkl_vm_vmcmulbyconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcmulbyconj

        subroutine vzmulbyconj(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzmulbyconj:mkl_vm_vzmulbyconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzmulbyconj

        subroutine vmzmulbyconj(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzmulbyconj:mkl_vm_vmzmulbyconj_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzmulbyconj


        subroutine vsnearbyint(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsnearbyint:mkl_vm_vsnearbyint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsnearbyint

        subroutine vmsnearbyint(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsnearbyint:mkl_vm_vmsnearbyint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsnearbyint

        subroutine vdnearbyint(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdnearbyint:mkl_vm_vdnearbyint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdnearbyint

        subroutine vmdnearbyint(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdnearbyint:mkl_vm_vmdnearbyint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdnearbyint


        subroutine vsnextafter(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsnextafter:mkl_vm_vsnextafter_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsnextafter

        subroutine vmsnextafter(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsnextafter:mkl_vm_vmsnextafter_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsnextafter

        subroutine vdnextafter(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdnextafter:mkl_vm_vdnextafter_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdnextafter

        subroutine vmdnextafter(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdnextafter:mkl_vm_vmdnextafter_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdnextafter


        subroutine vcpow(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcpow:mkl_vm_vcpow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcpow

        subroutine vmcpow(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcpow:mkl_vm_vmcpow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcpow

        subroutine vzpow(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzpow:mkl_vm_vzpow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzpow

        subroutine vmzpow(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzpow:mkl_vm_vmzpow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzpow

        subroutine vspow(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vspow:mkl_vm_vspow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vspow

        subroutine vmspow(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmspow:mkl_vm_vmspow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmspow

        subroutine vdpow(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdpow:mkl_vm_vdpow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdpow

        subroutine vmdpow(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdpow:mkl_vm_vmdpow_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdpow


        subroutine vspow2o3(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vspow2o3:mkl_vm_vspow2o3_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vspow2o3

        subroutine vmspow2o3(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmspow2o3:mkl_vm_vmspow2o3_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmspow2o3

        subroutine vdpow2o3(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdpow2o3:mkl_vm_vdpow2o3_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdpow2o3

        subroutine vmdpow2o3(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdpow2o3:mkl_vm_vmdpow2o3_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdpow2o3


        subroutine vspow3o2(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vspow3o2:mkl_vm_vspow3o2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vspow3o2

        subroutine vmspow3o2(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmspow3o2:mkl_vm_vmspow3o2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmspow3o2

        subroutine vdpow3o2(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdpow3o2:mkl_vm_vdpow3o2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdpow3o2

        subroutine vmdpow3o2(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdpow3o2:mkl_vm_vmdpow3o2_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdpow3o2


        subroutine vspowr(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vspowr:mkl_vm_vspowr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vspowr

        subroutine vmspowr(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmspowr:mkl_vm_vmspowr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmspowr

        subroutine vdpowr(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdpowr:mkl_vm_vdpowr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdpowr

        subroutine vmdpowr(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdpowr:mkl_vm_vmdpowr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdpowr


        subroutine vcpowx(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcpowx:mkl_vm_vcpowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcpowx

        subroutine vmcpowx(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcpowx:mkl_vm_vmcpowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcpowx

        subroutine vzpowx(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzpowx:mkl_vm_vzpowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzpowx

        subroutine vmzpowx(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzpowx:mkl_vm_vmzpowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzpowx

        subroutine vspowx(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vspowx:mkl_vm_vspowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vspowx

        subroutine vmspowx(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmspowx:mkl_vm_vmspowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmspowx

        subroutine vdpowx(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdpowx:mkl_vm_vdpowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdpowx

        subroutine vmdpowx(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdpowx:mkl_vm_vmdpowx_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdpowx


        subroutine vsremainder(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsremainder:mkl_vm_vsremainder_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsremainder

        subroutine vmsremainder(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsremainder:mkl_vm_vmsremainder_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsremainder

        subroutine vdremainder(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdremainder:mkl_vm_vdremainder_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdremainder

        subroutine vmdremainder(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdremainder:mkl_vm_vmdremainder_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdremainder


        subroutine vsrint(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsrint:mkl_vm_vsrint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsrint

        subroutine vmsrint(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsrint:mkl_vm_vmsrint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsrint

        subroutine vdrint(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdrint:mkl_vm_vdrint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdrint

        subroutine vmdrint(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdrint:mkl_vm_vmdrint_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdrint


        subroutine vsround(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vsround:mkl_vm_vsround_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vsround

        subroutine vmsround(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmsround:mkl_vm_vmsround_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmsround

        subroutine vdround(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdround:mkl_vm_vdround_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdround

        subroutine vmdround(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdround:mkl_vm_vmdround_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdround


        subroutine vcsin(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcsin:mkl_vm_vcsin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcsin

        subroutine vmcsin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcsin:mkl_vm_vmcsin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcsin

        subroutine vzsin(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzsin:mkl_vm_vzsin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzsin

        subroutine vmzsin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzsin:mkl_vm_vmzsin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzsin

        subroutine vssin(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssin:mkl_vm_vssin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssin

        subroutine vmssin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssin:mkl_vm_vmssin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssin

        subroutine vdsin(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsin:mkl_vm_vdsin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsin

        subroutine vmdsin(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsin:mkl_vm_vmdsin_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsin


        subroutine vssincos(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
!$omp declare variant ( vssincos:mkl_vm_vssincos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssincos

        subroutine vmssincos(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssincos:mkl_vm_vmssincos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssincos

        subroutine vdsincos(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
!$omp declare variant ( vdsincos:mkl_vm_vdsincos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsincos

        subroutine vmdsincos(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsincos:mkl_vm_vmdsincos_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsincos


        subroutine vssind(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssind:mkl_vm_vssind_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssind

        subroutine vmssind(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssind:mkl_vm_vmssind_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssind

        subroutine vdsind(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsind:mkl_vm_vdsind_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsind

        subroutine vmdsind(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsind:mkl_vm_vmdsind_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsind


        subroutine vcsinh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcsinh:mkl_vm_vcsinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcsinh

        subroutine vmcsinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcsinh:mkl_vm_vmcsinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcsinh

        subroutine vzsinh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzsinh:mkl_vm_vzsinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzsinh

        subroutine vmzsinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzsinh:mkl_vm_vmzsinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzsinh

        subroutine vssinh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssinh:mkl_vm_vssinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssinh

        subroutine vmssinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssinh:mkl_vm_vmssinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssinh

        subroutine vdsinh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsinh:mkl_vm_vdsinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsinh

        subroutine vmdsinh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsinh:mkl_vm_vmdsinh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsinh


        subroutine vssinpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssinpi:mkl_vm_vssinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssinpi

        subroutine vmssinpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssinpi:mkl_vm_vmssinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssinpi

        subroutine vdsinpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsinpi:mkl_vm_vdsinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsinpi

        subroutine vmdsinpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsinpi:mkl_vm_vmdsinpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsinpi


        subroutine vssqr(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssqr:mkl_vm_vssqr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssqr

        subroutine vmssqr(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssqr:mkl_vm_vmssqr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssqr

        subroutine vdsqr(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsqr:mkl_vm_vdsqr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsqr

        subroutine vmdsqr(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsqr:mkl_vm_vmdsqr_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsqr


        subroutine vcsqrt(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcsqrt:mkl_vm_vcsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcsqrt

        subroutine vmcsqrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcsqrt:mkl_vm_vmcsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcsqrt

        subroutine vzsqrt(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzsqrt:mkl_vm_vzsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzsqrt

        subroutine vmzsqrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzsqrt:mkl_vm_vmzsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzsqrt

        subroutine vssqrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssqrt:mkl_vm_vssqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssqrt

        subroutine vmssqrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssqrt:mkl_vm_vmssqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssqrt

        subroutine vdsqrt(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsqrt:mkl_vm_vdsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsqrt

        subroutine vmdsqrt(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsqrt:mkl_vm_vmdsqrt_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsqrt


        subroutine vcsub(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vcsub:mkl_vm_vcsub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vcsub

        subroutine vmcsub(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmcsub:mkl_vm_vmcsub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmcsub

        subroutine vzsub(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vzsub:mkl_vm_vzsub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vzsub

        subroutine vmzsub(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmzsub:mkl_vm_vmzsub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmzsub

        subroutine vssub(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vssub:mkl_vm_vssub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vssub

        subroutine vmssub(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmssub:mkl_vm_vmssub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmssub

        subroutine vdsub(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdsub:mkl_vm_vdsub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdsub

        subroutine vmdsub(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdsub:mkl_vm_vmdsub_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdsub


        subroutine vctan(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vctan:mkl_vm_vctan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vctan

        subroutine vmctan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmctan:mkl_vm_vmctan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmctan

        subroutine vztan(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vztan:mkl_vm_vztan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vztan

        subroutine vmztan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmztan:mkl_vm_vmztan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmztan

        subroutine vstan(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vstan:mkl_vm_vstan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vstan

        subroutine vmstan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmstan:mkl_vm_vmstan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmstan

        subroutine vdtan(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdtan:mkl_vm_vdtan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdtan

        subroutine vmdtan(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdtan:mkl_vm_vmdtan_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdtan


        subroutine vstand(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vstand:mkl_vm_vstand_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vstand

        subroutine vmstand(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmstand:mkl_vm_vmstand_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmstand

        subroutine vdtand(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdtand:mkl_vm_vdtand_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdtand

        subroutine vmdtand(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdtand:mkl_vm_vmdtand_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdtand


        subroutine vctanh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vctanh:mkl_vm_vctanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vctanh

        subroutine vmctanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmctanh:mkl_vm_vmctanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmctanh

        subroutine vztanh(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vztanh:mkl_vm_vztanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vztanh

        subroutine vmztanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmztanh:mkl_vm_vmztanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmztanh

        subroutine vstanh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vstanh:mkl_vm_vstanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vstanh

        subroutine vmstanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmstanh:mkl_vm_vmstanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmstanh

        subroutine vdtanh(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdtanh:mkl_vm_vdtanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdtanh

        subroutine vmdtanh(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdtanh:mkl_vm_vmdtanh_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdtanh


        subroutine vstanpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vstanpi:mkl_vm_vstanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vstanpi

        subroutine vmstanpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmstanpi:mkl_vm_vmstanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmstanpi

        subroutine vdtanpi(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdtanpi:mkl_vm_vdtanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdtanpi

        subroutine vmdtanpi(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdtanpi:mkl_vm_vmdtanpi_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdtanpi


        subroutine vstgamma(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vstgamma:mkl_vm_vstgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vstgamma

        subroutine vmstgamma(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmstgamma:mkl_vm_vmstgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmstgamma

        subroutine vdtgamma(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdtgamma:mkl_vm_vdtgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdtgamma

        subroutine vmdtgamma(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdtgamma:mkl_vm_vmdtgamma_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdtgamma


        subroutine vstrunc(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
!$omp declare variant ( vstrunc:mkl_vm_vstrunc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vstrunc

        subroutine vmstrunc(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmstrunc:mkl_vm_vmstrunc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmstrunc

        subroutine vdtrunc(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
!$omp declare variant ( vdtrunc:mkl_vm_vdtrunc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vdtrunc

        subroutine vmdtrunc(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
!$omp declare variant ( vmdtrunc:mkl_vm_vmdtrunc_omp_offload_ilp64 ) match( construct={target variant dispatch}, device={arch(gen)} )
        end subroutine vmdtrunc



    end interface

end module
