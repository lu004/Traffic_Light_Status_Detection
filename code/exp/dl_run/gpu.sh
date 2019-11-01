#!/bin/bash
while :
do
	nvidia-smi --query-gpu=gpu_name,memory.used,utilization.gpu --format=csv >> gpu.log 
	sleep 1
	echo 
done
