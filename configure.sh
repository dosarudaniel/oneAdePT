#!/bin/bash
#This script sets the environment variables needed to compile and run the tests
export DPCPP_HOME=~/sycl_workspace
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
export SYCL_ROOT_DIR=~/sycl_workspace/llvm/build
export CC=$SYCL_ROOT/bin/clang
export CXX=$SYCL_ROOT/bin/clang++
