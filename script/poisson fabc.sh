#!/bin/bash
#
#
# This Script is to train poisson with
# different training points to plot
# figure.2 A and B 、C
#
# # # # # # # # # # # # # # # # # # # # # #


# pinn
# for id in {1..10..1}
#   do
#     python ../../poisson\ 1d.py --net_type "pinn"
#     echo "pinn" "$id" "-" "$Nx" " completed!"
#   done

# gpinn w=1 
# for id in {1..10..1}
#   do
#     python ../../poisson\ 1d.py --net_type "gpinn" --g_weight 1
#     echo "gpinn, w = 1" "$id" "-" "$Nx" " completed!"
#   done

# gpinn w=0.01
for id in {1..10..1}
  do
    python ../../poisson\ 1d.py --net_type "gpinn" --g_weight 0.01
    echo "gpinn, w = 0.01" "$id" "-" "$Nx" " completed!"
  done