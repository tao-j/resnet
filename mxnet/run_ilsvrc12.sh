time_tag=`date +"%m-%d_%H:%M:%S"`
log_folder=log_ilsvrc12

for i in `seq 8 8`
do
log_file=$log_folder/output_$i_$time_tag.log
stat_file=$log_folder/gpustat_$i_$time_tag.log
    cat resnet_ilsvrc12*.py > $log_file
    (time python resnet_ilsvrc12_main.py --gpus 0,1,2,3,4,5,6,7 --batch-size 256 --data-dir ./data_ilsvrc12/) 2>&1 | tee -a $log_file &
    sleep 30; nvidia-smi 2>&1 | tee $stat_file
    sleep 30; nvidia-smi 2>&1 | tee -a $stat_file
    wait
done
