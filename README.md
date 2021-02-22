Step 0 Get the llvm sycl branch:   
`export DPCPP_HOME=~/sycl_workspace`   
`mkdir $DPCPP_HOME`   
`cd $DPCPP_HOME`   
`git clone https://github.com/intel/llvm -b sycl`   
`python $DPCPP_HOME/llvm/buildbot/configure.py --cuda`   
`python $DPCPP_HOME/llvm/buildbot/compile.py`   
   
Step 1 Set up the environment variables:     
`. configure.sh`    
   
Step 2 Configure cmake:   
`mkdir build && cd build`         
`cmake ../ -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++`   
   
Step 3 To compile:   
`make test4`   
   
Step 4 To run test4:   
`SYCL_DEVICE_FILTER=PI_LEVEL_ZERO ./test4`  
or    
`SYCL_DEVICE_FILTER=PI_CUDA ./test4`  

