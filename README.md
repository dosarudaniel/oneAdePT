Devcloud setup: 
  
Step 1 Set up the environment variables:     
`export SYCL_ROOT_DIR=/glob/development-tools/versions/oneapi/gold/inteloneapi/compiler/2021.1.2/linux/`  

Step 2 Configure cmake:   
`mkdir build && cd build`         
`cmake ../ -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/dpcpp`   
   
Step 3 To compile:   
`make test8`   
   
Step 4 To run test8:   
`qsub -l nodes=1:gpu:ppn=2 -d . run8.sh`

