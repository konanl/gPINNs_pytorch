#!/bin/bash
#
#
# This Script is to train function with
# different training points to plot figure
# 1 A and B
#
# # # # # # # # # # # # # # # # # # # # # #\

for net_type in 'gpinn' 'pinn'
    do
    for id in {1..10..1}
      do
          python ../../function.py --net_type $net_type 
      done
    done
