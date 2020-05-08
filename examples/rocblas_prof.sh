#!/bin/sh

if [ $# -le 0 ]; then
	echo "Usage: ./rocblas_prof.sh <example.py>"
	exit 1
fi

FNAME=${1%.*}
LOG_DIR=out/$FNAME

mkdir -p $LOG_DIR

export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=$LOG_DIR/$FNAME.csv

python3.6 $1 && cat $ROCBLAS_LOG_BENCH_PATH
