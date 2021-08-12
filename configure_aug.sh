#!/bin/bash
#This script sets the environment variables needed to compile and run the tests
export DPCPP_HOME=~/sycl_workspace_aug
export PATH=$DPCPP_HOME/llvm/build/install/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/install/lib:$LD_LIBRARY_PATH
export SYCL_ROOT_DIR=$DPCPP_HOME/llvm/build/install
export CC=$SYCL_ROOT_DIR/bin/clang
export CXX=$SYCL_ROOT_DIR/bin/clang++
