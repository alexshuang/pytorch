#!/bin/sh

apt update && apt install vim git ca-certificates wget cmake -y
apt install rock-dkms rocm-dev rocm-libs miopen-hip hipsparse rccl -y

wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh 
./Anaconda3-2020.02-Linux-x86_64.sh -b
echo "PATH=/root/anaconda3/bin:$PATH" >> ~/.bashrc
. ~/.bashrc

git clone https://github.com/alexshuang/pytorch.git -b enable_HgemmBatched
cd pytorch
git submodule update --init --recursive

export PYTORCH_ROCM_ARCH=gfx908

# python environment
conda create --name pytorch python=3.6
conda init bash
conda activate pytorch

pip install enum34 numpy pyyaml setuptools typing cffi future hypothesis
mv ~/anaconda3/envs/pytorch/compiler_compat/ld ~/anaconda3/envs/pytorch/compiler_compat/ld.bak

# hipfiy
python tools/amd_build/build_amd.py 

# do compile
export USE_NINJA=1
export hip_DIR=/opt/rocm/hip/lib/cmake/hip
export hcc_DIR=/opt/rocm/hcc/lib/cmake/hcc/
export amd_comgr_DIR=/opt/rocm/lib/cmake/amd_comgr/
export RCCL_DIR=/opt/rocm/rccl/lib/cmake/rccl/
USE_ROCM=1 USE_LMDB=1 BUILD_CAFFE2_OPS=0 BUILD_TEST=0 USE_OPENCV=1 MAX_JOBS=`nproc` python setup.py install

# install fairseq for transfomer benchmark
pip install fairseq
