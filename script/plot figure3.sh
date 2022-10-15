#!/bin/bash
#
#
# This Script is to train function with
# different training points to plot
# figure 3
#
# # # # # # # # # # # # # # # # # # # # # #

# for id in {1..10..1}
#   do
#     python ../../diffusion\ reaction.py --net_type "pinn"
#     echo "pinn" "$id"  " completed!"
#   done

# for id in {1..10..1}
#   do
#     python ../../diffusion\ reaction.py --net_type "gpinn" --g_weight 1
#     echo "gpinn, w = 1" "$id"  " completed!"
#   done

# for id in {1..10..1}
#   do
#     python ../../diffusion\ reaction.py --net_type "gpinn" --g_weight 0.1
#     echo "gpinn, w = 0.1" "$id" " completed!"
#   done

for id in {1..10..1}
  do
    python ../../diffusion\ reaction.py --net_type "gpinn" --g_weight 0.01
    echo "gpinn, w = 0.01" "$id"  " completed!"
  done