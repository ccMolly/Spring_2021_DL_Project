while :
do
	nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv >> gpu_utillization.log
	sleep 1
done
