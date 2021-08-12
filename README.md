Step 0 Get the llvm sycl branch:   
`export DPCPP_HOME=~/sycl_workspace_aug`   
`mkdir $DPCPP_HOME`   
`cd $DPCPP_HOME`   
`git clone https://github.com/intel/llvm -b sycl`   
`git checkout e8f2166fdd09ce3f6a15bcb7b5783cbbd149c0f7`
`python $DPCPP_HOME/llvm/buildbot/configure.py --cuda`   
`python $DPCPP_HOME/llvm/buildbot/compile.py`   
`cd llvm/build`
`ninja install`
   
Step 1 Set up the environment variables:     
`. configure.sh`    
   
Step 2 Configure cmake:   
`mkdir build && cd build`         
`cmake ../ -DCMAKE_INSTALL_PREFIX="~/local" -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++`   
   
Step 3 To compile:   
`make`   
   
Step 4 To run test1:   
`SYCL_DEVICE_FILTER=PI_LEVEL_ZERO ./test1`  
or    
`SYCL_DEVICE_FILTER=PI_CUDA ./test1`  

