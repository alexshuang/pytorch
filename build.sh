#!/bin/sh

export PYTORCH_ROCM_ARCH=gfx908

# do compile
export USE_NINJA=1
export hip_DIR=/opt/rocm/hip/lib/cmake/hip
export hcc_DIR=/opt/rocm/hcc/lib/cmake/hcc/
export amd_comgr_DIR=/opt/rocm/lib/cmake/amd_comgr/
export RCCL_DIR=/opt/rocm/rccl/lib/cmake/rccl/

#python setup.py clean
#DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_ROCM=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 BUILD_CAFFE2_OPS=0 USE_OPENMP=0 MAX_JOBS=`nproc` python setup.py develop
#DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_ROCM=1 BUILD_TEST=1 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 BUILD_CAFFE2_OPS=0 USE_OPENMP=0 MAX_JOBS=`nproc` python setup.py develop
USE_ROCM=1 USE_LMDB=1 BUILD_CAFFE2_OPS=0 BUILD_TEST=1 DEBUG=1 USE_OPENCV=1 MAX_JOBS=`nproc` python3.6 setup.py develop
