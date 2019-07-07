#!/bin/bash
####################################
#
# tensorflow 
#
####################################

docker run --runtime=nvidia -it -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.12.3-gpu-py3 bash
