#!/bin/bash
####################################
#
# tensorflow 
#
####################################

docker run --runtime=nvidia -it -p 8888:8888 -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.12.3-gpu-py3
