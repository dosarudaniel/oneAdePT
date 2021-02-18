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


    interface
        integer(kind=8) function mkl_vm_vmlsetmode_omp_offload_ilp64(n)
        integer(kind=8), intent(in) :: n
        end function mkl_vm_vmlsetmode_omp_offload_ilp64

        subroutine mkl_vm_vcabs_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcabs_omp_offload_ilp64

        subroutine mkl_vm_vmcabs_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcabs_omp_offload_ilp64

        subroutine mkl_vm_vzabs_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzabs_omp_offload_ilp64

        subroutine mkl_vm_vmzabs_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzabs_omp_offload_ilp64

        subroutine mkl_vm_vsabs_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsabs_omp_offload_ilp64

        subroutine mkl_vm_vmsabs_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsabs_omp_offload_ilp64

        subroutine mkl_vm_vdabs_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdabs_omp_offload_ilp64

        subroutine mkl_vm_vmdabs_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdabs_omp_offload_ilp64


        subroutine mkl_vm_vcacos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcacos_omp_offload_ilp64

        subroutine mkl_vm_vmcacos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcacos_omp_offload_ilp64

        subroutine mkl_vm_vzacos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzacos_omp_offload_ilp64

        subroutine mkl_vm_vmzacos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzacos_omp_offload_ilp64

        subroutine mkl_vm_vsacos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsacos_omp_offload_ilp64

        subroutine mkl_vm_vmsacos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsacos_omp_offload_ilp64

        subroutine mkl_vm_vdacos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdacos_omp_offload_ilp64

        subroutine mkl_vm_vmdacos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdacos_omp_offload_ilp64


        subroutine mkl_vm_vcacosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcacosh_omp_offload_ilp64

        subroutine mkl_vm_vmcacosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcacosh_omp_offload_ilp64

        subroutine mkl_vm_vzacosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzacosh_omp_offload_ilp64

        subroutine mkl_vm_vmzacosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzacosh_omp_offload_ilp64

        subroutine mkl_vm_vsacosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsacosh_omp_offload_ilp64

        subroutine mkl_vm_vmsacosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsacosh_omp_offload_ilp64

        subroutine mkl_vm_vdacosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdacosh_omp_offload_ilp64

        subroutine mkl_vm_vmdacosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdacosh_omp_offload_ilp64


        subroutine mkl_vm_vsacospi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsacospi_omp_offload_ilp64

        subroutine mkl_vm_vmsacospi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsacospi_omp_offload_ilp64

        subroutine mkl_vm_vdacospi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdacospi_omp_offload_ilp64

        subroutine mkl_vm_vmdacospi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdacospi_omp_offload_ilp64


        subroutine mkl_vm_vcadd_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcadd_omp_offload_ilp64

        subroutine mkl_vm_vmcadd_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcadd_omp_offload_ilp64

        subroutine mkl_vm_vzadd_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzadd_omp_offload_ilp64

        subroutine mkl_vm_vmzadd_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzadd_omp_offload_ilp64

        subroutine mkl_vm_vsadd_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsadd_omp_offload_ilp64

        subroutine mkl_vm_vmsadd_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsadd_omp_offload_ilp64

        subroutine mkl_vm_vdadd_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdadd_omp_offload_ilp64

        subroutine mkl_vm_vmdadd_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdadd_omp_offload_ilp64


        subroutine mkl_vm_vcarg_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcarg_omp_offload_ilp64

        subroutine mkl_vm_vmcarg_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcarg_omp_offload_ilp64

        subroutine mkl_vm_vzarg_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzarg_omp_offload_ilp64

        subroutine mkl_vm_vmzarg_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzarg_omp_offload_ilp64


        subroutine mkl_vm_vcasin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcasin_omp_offload_ilp64

        subroutine mkl_vm_vmcasin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcasin_omp_offload_ilp64

        subroutine mkl_vm_vzasin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzasin_omp_offload_ilp64

        subroutine mkl_vm_vmzasin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzasin_omp_offload_ilp64

        subroutine mkl_vm_vsasin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsasin_omp_offload_ilp64

        subroutine mkl_vm_vmsasin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsasin_omp_offload_ilp64

        subroutine mkl_vm_vdasin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdasin_omp_offload_ilp64

        subroutine mkl_vm_vmdasin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdasin_omp_offload_ilp64


        subroutine mkl_vm_vcasinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcasinh_omp_offload_ilp64

        subroutine mkl_vm_vmcasinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcasinh_omp_offload_ilp64

        subroutine mkl_vm_vzasinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzasinh_omp_offload_ilp64

        subroutine mkl_vm_vmzasinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzasinh_omp_offload_ilp64

        subroutine mkl_vm_vsasinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsasinh_omp_offload_ilp64

        subroutine mkl_vm_vmsasinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsasinh_omp_offload_ilp64

        subroutine mkl_vm_vdasinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdasinh_omp_offload_ilp64

        subroutine mkl_vm_vmdasinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdasinh_omp_offload_ilp64


        subroutine mkl_vm_vsasinpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsasinpi_omp_offload_ilp64

        subroutine mkl_vm_vmsasinpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsasinpi_omp_offload_ilp64

        subroutine mkl_vm_vdasinpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdasinpi_omp_offload_ilp64

        subroutine mkl_vm_vmdasinpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdasinpi_omp_offload_ilp64


        subroutine mkl_vm_vcatan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcatan_omp_offload_ilp64

        subroutine mkl_vm_vmcatan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcatan_omp_offload_ilp64

        subroutine mkl_vm_vzatan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzatan_omp_offload_ilp64

        subroutine mkl_vm_vmzatan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzatan_omp_offload_ilp64

        subroutine mkl_vm_vsatan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsatan_omp_offload_ilp64

        subroutine mkl_vm_vmsatan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsatan_omp_offload_ilp64

        subroutine mkl_vm_vdatan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdatan_omp_offload_ilp64

        subroutine mkl_vm_vmdatan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdatan_omp_offload_ilp64


        subroutine mkl_vm_vsatan2_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsatan2_omp_offload_ilp64

        subroutine mkl_vm_vmsatan2_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsatan2_omp_offload_ilp64

        subroutine mkl_vm_vdatan2_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdatan2_omp_offload_ilp64

        subroutine mkl_vm_vmdatan2_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdatan2_omp_offload_ilp64


        subroutine mkl_vm_vsatan2pi_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsatan2pi_omp_offload_ilp64

        subroutine mkl_vm_vmsatan2pi_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsatan2pi_omp_offload_ilp64

        subroutine mkl_vm_vdatan2pi_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdatan2pi_omp_offload_ilp64

        subroutine mkl_vm_vmdatan2pi_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdatan2pi_omp_offload_ilp64


        subroutine mkl_vm_vcatanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcatanh_omp_offload_ilp64

        subroutine mkl_vm_vmcatanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcatanh_omp_offload_ilp64

        subroutine mkl_vm_vzatanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzatanh_omp_offload_ilp64

        subroutine mkl_vm_vmzatanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzatanh_omp_offload_ilp64

        subroutine mkl_vm_vsatanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsatanh_omp_offload_ilp64

        subroutine mkl_vm_vmsatanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsatanh_omp_offload_ilp64

        subroutine mkl_vm_vdatanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdatanh_omp_offload_ilp64

        subroutine mkl_vm_vmdatanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdatanh_omp_offload_ilp64


        subroutine mkl_vm_vsatanpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsatanpi_omp_offload_ilp64

        subroutine mkl_vm_vmsatanpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsatanpi_omp_offload_ilp64

        subroutine mkl_vm_vdatanpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdatanpi_omp_offload_ilp64

        subroutine mkl_vm_vmdatanpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdatanpi_omp_offload_ilp64


        subroutine mkl_vm_vscbrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscbrt_omp_offload_ilp64

        subroutine mkl_vm_vmscbrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscbrt_omp_offload_ilp64

        subroutine mkl_vm_vdcbrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcbrt_omp_offload_ilp64

        subroutine mkl_vm_vmdcbrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcbrt_omp_offload_ilp64


        subroutine mkl_vm_vscdfnorm_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscdfnorm_omp_offload_ilp64

        subroutine mkl_vm_vmscdfnorm_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscdfnorm_omp_offload_ilp64

        subroutine mkl_vm_vdcdfnorm_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcdfnorm_omp_offload_ilp64

        subroutine mkl_vm_vmdcdfnorm_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcdfnorm_omp_offload_ilp64


        subroutine mkl_vm_vscdfnorminv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscdfnorminv_omp_offload_ilp64

        subroutine mkl_vm_vmscdfnorminv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscdfnorminv_omp_offload_ilp64

        subroutine mkl_vm_vdcdfnorminv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcdfnorminv_omp_offload_ilp64

        subroutine mkl_vm_vmdcdfnorminv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcdfnorminv_omp_offload_ilp64


        subroutine mkl_vm_vsceil_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsceil_omp_offload_ilp64

        subroutine mkl_vm_vmsceil_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsceil_omp_offload_ilp64

        subroutine mkl_vm_vdceil_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdceil_omp_offload_ilp64

        subroutine mkl_vm_vmdceil_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdceil_omp_offload_ilp64


        subroutine mkl_vm_vccis_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vccis_omp_offload_ilp64

        subroutine mkl_vm_vmccis_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmccis_omp_offload_ilp64

        subroutine mkl_vm_vzcis_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzcis_omp_offload_ilp64

        subroutine mkl_vm_vmzcis_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzcis_omp_offload_ilp64


        subroutine mkl_vm_vcconj_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcconj_omp_offload_ilp64

        subroutine mkl_vm_vmcconj_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcconj_omp_offload_ilp64

        subroutine mkl_vm_vzconj_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzconj_omp_offload_ilp64

        subroutine mkl_vm_vmzconj_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzconj_omp_offload_ilp64


        subroutine mkl_vm_vscopysign_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscopysign_omp_offload_ilp64

        subroutine mkl_vm_vmscopysign_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscopysign_omp_offload_ilp64

        subroutine mkl_vm_vdcopysign_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcopysign_omp_offload_ilp64

        subroutine mkl_vm_vmdcopysign_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcopysign_omp_offload_ilp64


        subroutine mkl_vm_vccos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vccos_omp_offload_ilp64

        subroutine mkl_vm_vmccos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmccos_omp_offload_ilp64

        subroutine mkl_vm_vzcos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzcos_omp_offload_ilp64

        subroutine mkl_vm_vmzcos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzcos_omp_offload_ilp64

        subroutine mkl_vm_vscos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscos_omp_offload_ilp64

        subroutine mkl_vm_vmscos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscos_omp_offload_ilp64

        subroutine mkl_vm_vdcos_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcos_omp_offload_ilp64

        subroutine mkl_vm_vmdcos_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcos_omp_offload_ilp64


        subroutine mkl_vm_vscosd_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscosd_omp_offload_ilp64

        subroutine mkl_vm_vmscosd_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscosd_omp_offload_ilp64

        subroutine mkl_vm_vdcosd_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcosd_omp_offload_ilp64

        subroutine mkl_vm_vmdcosd_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcosd_omp_offload_ilp64


        subroutine mkl_vm_vccosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vccosh_omp_offload_ilp64

        subroutine mkl_vm_vmccosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmccosh_omp_offload_ilp64

        subroutine mkl_vm_vzcosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzcosh_omp_offload_ilp64

        subroutine mkl_vm_vmzcosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzcosh_omp_offload_ilp64

        subroutine mkl_vm_vscosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscosh_omp_offload_ilp64

        subroutine mkl_vm_vmscosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscosh_omp_offload_ilp64

        subroutine mkl_vm_vdcosh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcosh_omp_offload_ilp64

        subroutine mkl_vm_vmdcosh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcosh_omp_offload_ilp64


        subroutine mkl_vm_vscospi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vscospi_omp_offload_ilp64

        subroutine mkl_vm_vmscospi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmscospi_omp_offload_ilp64

        subroutine mkl_vm_vdcospi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdcospi_omp_offload_ilp64

        subroutine mkl_vm_vmdcospi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdcospi_omp_offload_ilp64


        subroutine mkl_vm_vcdiv_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcdiv_omp_offload_ilp64

        subroutine mkl_vm_vmcdiv_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcdiv_omp_offload_ilp64

        subroutine mkl_vm_vzdiv_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzdiv_omp_offload_ilp64

        subroutine mkl_vm_vmzdiv_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzdiv_omp_offload_ilp64

        subroutine mkl_vm_vsdiv_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsdiv_omp_offload_ilp64

        subroutine mkl_vm_vmsdiv_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsdiv_omp_offload_ilp64

        subroutine mkl_vm_vddiv_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vddiv_omp_offload_ilp64

        subroutine mkl_vm_vmddiv_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmddiv_omp_offload_ilp64


        subroutine mkl_vm_vserf_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vserf_omp_offload_ilp64

        subroutine mkl_vm_vmserf_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmserf_omp_offload_ilp64

        subroutine mkl_vm_vderf_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vderf_omp_offload_ilp64

        subroutine mkl_vm_vmderf_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmderf_omp_offload_ilp64


        subroutine mkl_vm_vserfc_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vserfc_omp_offload_ilp64

        subroutine mkl_vm_vmserfc_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmserfc_omp_offload_ilp64

        subroutine mkl_vm_vderfc_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vderfc_omp_offload_ilp64

        subroutine mkl_vm_vmderfc_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmderfc_omp_offload_ilp64


        subroutine mkl_vm_vserfcinv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vserfcinv_omp_offload_ilp64

        subroutine mkl_vm_vmserfcinv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmserfcinv_omp_offload_ilp64

        subroutine mkl_vm_vderfcinv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vderfcinv_omp_offload_ilp64

        subroutine mkl_vm_vmderfcinv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmderfcinv_omp_offload_ilp64


        subroutine mkl_vm_vserfinv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vserfinv_omp_offload_ilp64

        subroutine mkl_vm_vmserfinv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmserfinv_omp_offload_ilp64

        subroutine mkl_vm_vderfinv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vderfinv_omp_offload_ilp64

        subroutine mkl_vm_vmderfinv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmderfinv_omp_offload_ilp64


        subroutine mkl_vm_vcexp_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcexp_omp_offload_ilp64

        subroutine mkl_vm_vmcexp_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcexp_omp_offload_ilp64

        subroutine mkl_vm_vzexp_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzexp_omp_offload_ilp64

        subroutine mkl_vm_vmzexp_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzexp_omp_offload_ilp64

        subroutine mkl_vm_vsexp_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsexp_omp_offload_ilp64

        subroutine mkl_vm_vmsexp_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsexp_omp_offload_ilp64

        subroutine mkl_vm_vdexp_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdexp_omp_offload_ilp64

        subroutine mkl_vm_vmdexp_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdexp_omp_offload_ilp64


        subroutine mkl_vm_vsexp2_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsexp2_omp_offload_ilp64

        subroutine mkl_vm_vmsexp2_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsexp2_omp_offload_ilp64

        subroutine mkl_vm_vdexp2_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdexp2_omp_offload_ilp64

        subroutine mkl_vm_vmdexp2_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdexp2_omp_offload_ilp64


        subroutine mkl_vm_vsexp10_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsexp10_omp_offload_ilp64

        subroutine mkl_vm_vmsexp10_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsexp10_omp_offload_ilp64

        subroutine mkl_vm_vdexp10_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdexp10_omp_offload_ilp64

        subroutine mkl_vm_vmdexp10_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdexp10_omp_offload_ilp64


        subroutine mkl_vm_vsexpint1_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsexpint1_omp_offload_ilp64

        subroutine mkl_vm_vmsexpint1_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsexpint1_omp_offload_ilp64

        subroutine mkl_vm_vdexpint1_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdexpint1_omp_offload_ilp64

        subroutine mkl_vm_vmdexpint1_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdexpint1_omp_offload_ilp64


        subroutine mkl_vm_vsexpm1_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsexpm1_omp_offload_ilp64

        subroutine mkl_vm_vmsexpm1_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsexpm1_omp_offload_ilp64

        subroutine mkl_vm_vdexpm1_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdexpm1_omp_offload_ilp64

        subroutine mkl_vm_vmdexpm1_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdexpm1_omp_offload_ilp64


        subroutine mkl_vm_vsfdim_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsfdim_omp_offload_ilp64

        subroutine mkl_vm_vmsfdim_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsfdim_omp_offload_ilp64

        subroutine mkl_vm_vdfdim_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdfdim_omp_offload_ilp64

        subroutine mkl_vm_vmdfdim_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdfdim_omp_offload_ilp64


        subroutine mkl_vm_vsfloor_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsfloor_omp_offload_ilp64

        subroutine mkl_vm_vmsfloor_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsfloor_omp_offload_ilp64

        subroutine mkl_vm_vdfloor_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdfloor_omp_offload_ilp64

        subroutine mkl_vm_vmdfloor_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdfloor_omp_offload_ilp64


        subroutine mkl_vm_vsfmax_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsfmax_omp_offload_ilp64

        subroutine mkl_vm_vmsfmax_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsfmax_omp_offload_ilp64

        subroutine mkl_vm_vdfmax_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdfmax_omp_offload_ilp64

        subroutine mkl_vm_vmdfmax_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdfmax_omp_offload_ilp64


        subroutine mkl_vm_vsfmin_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsfmin_omp_offload_ilp64

        subroutine mkl_vm_vmsfmin_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsfmin_omp_offload_ilp64

        subroutine mkl_vm_vdfmin_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdfmin_omp_offload_ilp64

        subroutine mkl_vm_vmdfmin_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdfmin_omp_offload_ilp64


        subroutine mkl_vm_vsfmod_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsfmod_omp_offload_ilp64

        subroutine mkl_vm_vmsfmod_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsfmod_omp_offload_ilp64

        subroutine mkl_vm_vdfmod_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdfmod_omp_offload_ilp64

        subroutine mkl_vm_vmdfmod_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdfmod_omp_offload_ilp64


        subroutine mkl_vm_vsfrac_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsfrac_omp_offload_ilp64

        subroutine mkl_vm_vmsfrac_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsfrac_omp_offload_ilp64

        subroutine mkl_vm_vdfrac_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdfrac_omp_offload_ilp64

        subroutine mkl_vm_vmdfrac_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdfrac_omp_offload_ilp64


        subroutine mkl_vm_vshypot_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vshypot_omp_offload_ilp64

        subroutine mkl_vm_vmshypot_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmshypot_omp_offload_ilp64

        subroutine mkl_vm_vdhypot_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdhypot_omp_offload_ilp64

        subroutine mkl_vm_vmdhypot_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdhypot_omp_offload_ilp64


        subroutine mkl_vm_vsinv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsinv_omp_offload_ilp64

        subroutine mkl_vm_vmsinv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsinv_omp_offload_ilp64

        subroutine mkl_vm_vdinv_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdinv_omp_offload_ilp64

        subroutine mkl_vm_vmdinv_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdinv_omp_offload_ilp64


        subroutine mkl_vm_vsinvcbrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsinvcbrt_omp_offload_ilp64

        subroutine mkl_vm_vmsinvcbrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsinvcbrt_omp_offload_ilp64

        subroutine mkl_vm_vdinvcbrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdinvcbrt_omp_offload_ilp64

        subroutine mkl_vm_vmdinvcbrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdinvcbrt_omp_offload_ilp64


        subroutine mkl_vm_vsinvsqrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsinvsqrt_omp_offload_ilp64

        subroutine mkl_vm_vmsinvsqrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsinvsqrt_omp_offload_ilp64

        subroutine mkl_vm_vdinvsqrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdinvsqrt_omp_offload_ilp64

        subroutine mkl_vm_vmdinvsqrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdinvsqrt_omp_offload_ilp64


        subroutine mkl_vm_vslgamma_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vslgamma_omp_offload_ilp64

        subroutine mkl_vm_vmslgamma_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmslgamma_omp_offload_ilp64

        subroutine mkl_vm_vdlgamma_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdlgamma_omp_offload_ilp64

        subroutine mkl_vm_vmdlgamma_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdlgamma_omp_offload_ilp64


        subroutine mkl_vm_vslinearfrac_omp_offload_ilp64(n, a, b, c, d, e, f, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: c
        real(kind=4), intent(inout) :: d
        real(kind=4), intent(inout) :: e
        real(kind=4), intent(inout) :: f
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vslinearfrac_omp_offload_ilp64

        subroutine mkl_vm_vmslinearfrac_omp_offload_ilp64(n, a, b, c, d, e, f, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: c
        real(kind=4), intent(inout) :: d
        real(kind=4), intent(inout) :: e
        real(kind=4), intent(inout) :: f
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmslinearfrac_omp_offload_ilp64

        subroutine mkl_vm_vdlinearfrac_omp_offload_ilp64(n, a, b, c, d, e, f, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: c
        real(kind=8), intent(inout) :: d
        real(kind=8), intent(inout) :: e
        real(kind=8), intent(inout) :: f
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdlinearfrac_omp_offload_ilp64

        subroutine mkl_vm_vmdlinearfrac_omp_offload_ilp64(n, a, b, c, d, e, f, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: c
        real(kind=8), intent(inout) :: d
        real(kind=8), intent(inout) :: e
        real(kind=8), intent(inout) :: f
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdlinearfrac_omp_offload_ilp64


        subroutine mkl_vm_vcln_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcln_omp_offload_ilp64

        subroutine mkl_vm_vmcln_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcln_omp_offload_ilp64

        subroutine mkl_vm_vzln_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzln_omp_offload_ilp64

        subroutine mkl_vm_vmzln_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzln_omp_offload_ilp64

        subroutine mkl_vm_vsln_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsln_omp_offload_ilp64

        subroutine mkl_vm_vmsln_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsln_omp_offload_ilp64

        subroutine mkl_vm_vdln_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdln_omp_offload_ilp64

        subroutine mkl_vm_vmdln_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdln_omp_offload_ilp64


        subroutine mkl_vm_vslog2_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vslog2_omp_offload_ilp64

        subroutine mkl_vm_vmslog2_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmslog2_omp_offload_ilp64

        subroutine mkl_vm_vdlog2_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdlog2_omp_offload_ilp64

        subroutine mkl_vm_vmdlog2_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdlog2_omp_offload_ilp64


        subroutine mkl_vm_vslogb_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vslogb_omp_offload_ilp64

        subroutine mkl_vm_vmslogb_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmslogb_omp_offload_ilp64

        subroutine mkl_vm_vdlogb_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdlogb_omp_offload_ilp64

        subroutine mkl_vm_vmdlogb_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdlogb_omp_offload_ilp64


        subroutine mkl_vm_vclog10_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vclog10_omp_offload_ilp64

        subroutine mkl_vm_vmclog10_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmclog10_omp_offload_ilp64

        subroutine mkl_vm_vzlog10_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzlog10_omp_offload_ilp64

        subroutine mkl_vm_vmzlog10_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzlog10_omp_offload_ilp64

        subroutine mkl_vm_vslog10_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vslog10_omp_offload_ilp64

        subroutine mkl_vm_vmslog10_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmslog10_omp_offload_ilp64

        subroutine mkl_vm_vdlog10_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdlog10_omp_offload_ilp64

        subroutine mkl_vm_vmdlog10_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdlog10_omp_offload_ilp64


        subroutine mkl_vm_vslog1p_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vslog1p_omp_offload_ilp64

        subroutine mkl_vm_vmslog1p_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmslog1p_omp_offload_ilp64

        subroutine mkl_vm_vdlog1p_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdlog1p_omp_offload_ilp64

        subroutine mkl_vm_vmdlog1p_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdlog1p_omp_offload_ilp64


        subroutine mkl_vm_vsmaxmag_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsmaxmag_omp_offload_ilp64

        subroutine mkl_vm_vmsmaxmag_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsmaxmag_omp_offload_ilp64

        subroutine mkl_vm_vdmaxmag_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdmaxmag_omp_offload_ilp64

        subroutine mkl_vm_vmdmaxmag_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdmaxmag_omp_offload_ilp64


        subroutine mkl_vm_vsminmag_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsminmag_omp_offload_ilp64

        subroutine mkl_vm_vmsminmag_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsminmag_omp_offload_ilp64

        subroutine mkl_vm_vdminmag_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdminmag_omp_offload_ilp64

        subroutine mkl_vm_vmdminmag_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdminmag_omp_offload_ilp64


        subroutine mkl_vm_vsmodf_omp_offload_ilp64(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
        end subroutine mkl_vm_vsmodf_omp_offload_ilp64

        subroutine mkl_vm_vmsmodf_omp_offload_ilp64(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsmodf_omp_offload_ilp64

        subroutine mkl_vm_vdmodf_omp_offload_ilp64(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
        end subroutine mkl_vm_vdmodf_omp_offload_ilp64

        subroutine mkl_vm_vmdmodf_omp_offload_ilp64(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdmodf_omp_offload_ilp64


        subroutine mkl_vm_vcmul_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcmul_omp_offload_ilp64

        subroutine mkl_vm_vmcmul_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcmul_omp_offload_ilp64

        subroutine mkl_vm_vzmul_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzmul_omp_offload_ilp64

        subroutine mkl_vm_vmzmul_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzmul_omp_offload_ilp64

        subroutine mkl_vm_vsmul_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsmul_omp_offload_ilp64

        subroutine mkl_vm_vmsmul_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsmul_omp_offload_ilp64

        subroutine mkl_vm_vdmul_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdmul_omp_offload_ilp64

        subroutine mkl_vm_vmdmul_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdmul_omp_offload_ilp64


        subroutine mkl_vm_vcmulbyconj_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcmulbyconj_omp_offload_ilp64

        subroutine mkl_vm_vmcmulbyconj_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcmulbyconj_omp_offload_ilp64

        subroutine mkl_vm_vzmulbyconj_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzmulbyconj_omp_offload_ilp64

        subroutine mkl_vm_vmzmulbyconj_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzmulbyconj_omp_offload_ilp64


        subroutine mkl_vm_vsnearbyint_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsnearbyint_omp_offload_ilp64

        subroutine mkl_vm_vmsnearbyint_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsnearbyint_omp_offload_ilp64

        subroutine mkl_vm_vdnearbyint_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdnearbyint_omp_offload_ilp64

        subroutine mkl_vm_vmdnearbyint_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdnearbyint_omp_offload_ilp64


        subroutine mkl_vm_vsnextafter_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsnextafter_omp_offload_ilp64

        subroutine mkl_vm_vmsnextafter_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsnextafter_omp_offload_ilp64

        subroutine mkl_vm_vdnextafter_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdnextafter_omp_offload_ilp64

        subroutine mkl_vm_vmdnextafter_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdnextafter_omp_offload_ilp64


        subroutine mkl_vm_vcpow_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcpow_omp_offload_ilp64

        subroutine mkl_vm_vmcpow_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcpow_omp_offload_ilp64

        subroutine mkl_vm_vzpow_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzpow_omp_offload_ilp64

        subroutine mkl_vm_vmzpow_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzpow_omp_offload_ilp64

        subroutine mkl_vm_vspow_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vspow_omp_offload_ilp64

        subroutine mkl_vm_vmspow_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmspow_omp_offload_ilp64

        subroutine mkl_vm_vdpow_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdpow_omp_offload_ilp64

        subroutine mkl_vm_vmdpow_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdpow_omp_offload_ilp64


        subroutine mkl_vm_vspow2o3_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vspow2o3_omp_offload_ilp64

        subroutine mkl_vm_vmspow2o3_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmspow2o3_omp_offload_ilp64

        subroutine mkl_vm_vdpow2o3_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdpow2o3_omp_offload_ilp64

        subroutine mkl_vm_vmdpow2o3_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdpow2o3_omp_offload_ilp64


        subroutine mkl_vm_vspow3o2_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vspow3o2_omp_offload_ilp64

        subroutine mkl_vm_vmspow3o2_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmspow3o2_omp_offload_ilp64

        subroutine mkl_vm_vdpow3o2_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdpow3o2_omp_offload_ilp64

        subroutine mkl_vm_vmdpow3o2_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdpow3o2_omp_offload_ilp64


        subroutine mkl_vm_vspowr_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vspowr_omp_offload_ilp64

        subroutine mkl_vm_vmspowr_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmspowr_omp_offload_ilp64

        subroutine mkl_vm_vdpowr_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdpowr_omp_offload_ilp64

        subroutine mkl_vm_vmdpowr_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdpowr_omp_offload_ilp64


        subroutine mkl_vm_vcpowx_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcpowx_omp_offload_ilp64

        subroutine mkl_vm_vmcpowx_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcpowx_omp_offload_ilp64

        subroutine mkl_vm_vzpowx_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzpowx_omp_offload_ilp64

        subroutine mkl_vm_vmzpowx_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzpowx_omp_offload_ilp64

        subroutine mkl_vm_vspowx_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vspowx_omp_offload_ilp64

        subroutine mkl_vm_vmspowx_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmspowx_omp_offload_ilp64

        subroutine mkl_vm_vdpowx_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdpowx_omp_offload_ilp64

        subroutine mkl_vm_vmdpowx_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdpowx_omp_offload_ilp64


        subroutine mkl_vm_vsremainder_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsremainder_omp_offload_ilp64

        subroutine mkl_vm_vmsremainder_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsremainder_omp_offload_ilp64

        subroutine mkl_vm_vdremainder_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdremainder_omp_offload_ilp64

        subroutine mkl_vm_vmdremainder_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdremainder_omp_offload_ilp64


        subroutine mkl_vm_vsrint_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsrint_omp_offload_ilp64

        subroutine mkl_vm_vmsrint_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsrint_omp_offload_ilp64

        subroutine mkl_vm_vdrint_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdrint_omp_offload_ilp64

        subroutine mkl_vm_vmdrint_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdrint_omp_offload_ilp64


        subroutine mkl_vm_vsround_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vsround_omp_offload_ilp64

        subroutine mkl_vm_vmsround_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmsround_omp_offload_ilp64

        subroutine mkl_vm_vdround_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdround_omp_offload_ilp64

        subroutine mkl_vm_vmdround_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdround_omp_offload_ilp64


        subroutine mkl_vm_vcsin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcsin_omp_offload_ilp64

        subroutine mkl_vm_vmcsin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcsin_omp_offload_ilp64

        subroutine mkl_vm_vzsin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzsin_omp_offload_ilp64

        subroutine mkl_vm_vmzsin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzsin_omp_offload_ilp64

        subroutine mkl_vm_vssin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssin_omp_offload_ilp64

        subroutine mkl_vm_vmssin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssin_omp_offload_ilp64

        subroutine mkl_vm_vdsin_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsin_omp_offload_ilp64

        subroutine mkl_vm_vmdsin_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsin_omp_offload_ilp64


        subroutine mkl_vm_vssincos_omp_offload_ilp64(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
        end subroutine mkl_vm_vssincos_omp_offload_ilp64

        subroutine mkl_vm_vmssincos_omp_offload_ilp64(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        real(kind=4), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssincos_omp_offload_ilp64

        subroutine mkl_vm_vdsincos_omp_offload_ilp64(n, a, y, z)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
        end subroutine mkl_vm_vdsincos_omp_offload_ilp64

        subroutine mkl_vm_vmdsincos_omp_offload_ilp64(n, a, y, z, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        real(kind=8), intent(inout) :: z(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsincos_omp_offload_ilp64


        subroutine mkl_vm_vssind_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssind_omp_offload_ilp64

        subroutine mkl_vm_vmssind_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssind_omp_offload_ilp64

        subroutine mkl_vm_vdsind_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsind_omp_offload_ilp64

        subroutine mkl_vm_vmdsind_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsind_omp_offload_ilp64


        subroutine mkl_vm_vcsinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcsinh_omp_offload_ilp64

        subroutine mkl_vm_vmcsinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcsinh_omp_offload_ilp64

        subroutine mkl_vm_vzsinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzsinh_omp_offload_ilp64

        subroutine mkl_vm_vmzsinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzsinh_omp_offload_ilp64

        subroutine mkl_vm_vssinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssinh_omp_offload_ilp64

        subroutine mkl_vm_vmssinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssinh_omp_offload_ilp64

        subroutine mkl_vm_vdsinh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsinh_omp_offload_ilp64

        subroutine mkl_vm_vmdsinh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsinh_omp_offload_ilp64


        subroutine mkl_vm_vssinpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssinpi_omp_offload_ilp64

        subroutine mkl_vm_vmssinpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssinpi_omp_offload_ilp64

        subroutine mkl_vm_vdsinpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsinpi_omp_offload_ilp64

        subroutine mkl_vm_vmdsinpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsinpi_omp_offload_ilp64


        subroutine mkl_vm_vssqr_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssqr_omp_offload_ilp64

        subroutine mkl_vm_vmssqr_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssqr_omp_offload_ilp64

        subroutine mkl_vm_vdsqr_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsqr_omp_offload_ilp64

        subroutine mkl_vm_vmdsqr_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsqr_omp_offload_ilp64


        subroutine mkl_vm_vcsqrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcsqrt_omp_offload_ilp64

        subroutine mkl_vm_vmcsqrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcsqrt_omp_offload_ilp64

        subroutine mkl_vm_vzsqrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzsqrt_omp_offload_ilp64

        subroutine mkl_vm_vmzsqrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzsqrt_omp_offload_ilp64

        subroutine mkl_vm_vssqrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssqrt_omp_offload_ilp64

        subroutine mkl_vm_vmssqrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssqrt_omp_offload_ilp64

        subroutine mkl_vm_vdsqrt_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsqrt_omp_offload_ilp64

        subroutine mkl_vm_vmdsqrt_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsqrt_omp_offload_ilp64


        subroutine mkl_vm_vcsub_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vcsub_omp_offload_ilp64

        subroutine mkl_vm_vmcsub_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: b(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmcsub_omp_offload_ilp64

        subroutine mkl_vm_vzsub_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vzsub_omp_offload_ilp64

        subroutine mkl_vm_vmzsub_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: b(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmzsub_omp_offload_ilp64

        subroutine mkl_vm_vssub_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vssub_omp_offload_ilp64

        subroutine mkl_vm_vmssub_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: b(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmssub_omp_offload_ilp64

        subroutine mkl_vm_vdsub_omp_offload_ilp64(n, a, b, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdsub_omp_offload_ilp64

        subroutine mkl_vm_vmdsub_omp_offload_ilp64(n, a, b, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: b(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdsub_omp_offload_ilp64


        subroutine mkl_vm_vctan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vctan_omp_offload_ilp64

        subroutine mkl_vm_vmctan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmctan_omp_offload_ilp64

        subroutine mkl_vm_vztan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vztan_omp_offload_ilp64

        subroutine mkl_vm_vmztan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmztan_omp_offload_ilp64

        subroutine mkl_vm_vstan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vstan_omp_offload_ilp64

        subroutine mkl_vm_vmstan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmstan_omp_offload_ilp64

        subroutine mkl_vm_vdtan_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdtan_omp_offload_ilp64

        subroutine mkl_vm_vmdtan_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdtan_omp_offload_ilp64


        subroutine mkl_vm_vstand_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vstand_omp_offload_ilp64

        subroutine mkl_vm_vmstand_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmstand_omp_offload_ilp64

        subroutine mkl_vm_vdtand_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdtand_omp_offload_ilp64

        subroutine mkl_vm_vmdtand_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdtand_omp_offload_ilp64


        subroutine mkl_vm_vctanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vctanh_omp_offload_ilp64

        subroutine mkl_vm_vmctanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=4), intent(inout) :: a(*)
        complex(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmctanh_omp_offload_ilp64

        subroutine mkl_vm_vztanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vztanh_omp_offload_ilp64

        subroutine mkl_vm_vmztanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        complex(kind=8), intent(inout) :: a(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmztanh_omp_offload_ilp64

        subroutine mkl_vm_vstanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vstanh_omp_offload_ilp64

        subroutine mkl_vm_vmstanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmstanh_omp_offload_ilp64

        subroutine mkl_vm_vdtanh_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdtanh_omp_offload_ilp64

        subroutine mkl_vm_vmdtanh_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdtanh_omp_offload_ilp64


        subroutine mkl_vm_vstanpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vstanpi_omp_offload_ilp64

        subroutine mkl_vm_vmstanpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmstanpi_omp_offload_ilp64

        subroutine mkl_vm_vdtanpi_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdtanpi_omp_offload_ilp64

        subroutine mkl_vm_vmdtanpi_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdtanpi_omp_offload_ilp64


        subroutine mkl_vm_vstgamma_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vstgamma_omp_offload_ilp64

        subroutine mkl_vm_vmstgamma_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmstgamma_omp_offload_ilp64

        subroutine mkl_vm_vdtgamma_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdtgamma_omp_offload_ilp64

        subroutine mkl_vm_vmdtgamma_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdtgamma_omp_offload_ilp64


        subroutine mkl_vm_vstrunc_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        end subroutine mkl_vm_vstrunc_omp_offload_ilp64

        subroutine mkl_vm_vmstrunc_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=4), intent(inout) :: a(*)
        real(kind=4), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmstrunc_omp_offload_ilp64

        subroutine mkl_vm_vdtrunc_omp_offload_ilp64(n, a, y)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        end subroutine mkl_vm_vdtrunc_omp_offload_ilp64

        subroutine mkl_vm_vmdtrunc_omp_offload_ilp64(n, a, y, mode)
        integer(kind=8), intent(in) :: n
        real(kind=8), intent(inout) :: a(*)
        real(kind=8), intent(inout) :: y(*)
        integer(kind=8), intent(in) :: mode
        end subroutine mkl_vm_vmdtrunc_omp_offload_ilp64



    end interface
