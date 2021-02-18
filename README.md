Step 0 Get the llvm sycl branch:   
`export DPCPP_HOME=~/sycl_workspace`   
`mkdir $DPCPP_HOME`   
`cd $DPCPP_HOME`   
`git clone https://github.com/intel/llvm -b sycl`   
`python $DPCPP_HOME/llvm/buildbot/configure.py --cuda`   
`python $DPCPP_HOME/llvm/buildbot/compile.py`   
   
Step 1 Set up the environment variables:   
`export DPCPP_HOME=~/sycl_workspace`   
`export PATH=$DPCPP_HOME/llvm/build/bin:$PATH`   
`export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH`  
`export SYCL_ROOT_DIR=~/sycl_workspace/llvm/build`      
`export CC=clang++`      
`export CXX=$SYCL_ROOT/bin/clang++` 
   
Step 2 Configure cmake:   
`mkdir build && cd build`         
`cmake ../ -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++`   
   
Step 3 To compile:   
`make`   
   
Step 4 To run test1:   
`SYCL_DEVICE_FILTER=PI_LEVEL_ZERO ./test1`  
or    
`SYCL_DEVICE_FILTER=PI_CUDA ./test1`  

