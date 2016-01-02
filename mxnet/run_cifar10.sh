time_tag=`date +"%m-%d_%H:%M:%S"`
log_folder=log_cifar10

for i in `seq 1 1`
do
log_file=$log_folder/output_$i_$time_tag.log
stat_file=$log_folder/gpustat_$i_$time_tag.log
    cat resnet_cifar10*.py > $log_file
    (time python resnet_cifar10_main.py --gpus 1,5 --data-dir ./data_cifar10/ --batch-size 128) 2>&1 | tee -a $log_file &
    sleep 30; nvidia-smi 2>&1 | tee $stat_file
    sleep 30; nvidia-smi 2>&1 | tee -a $stat_file
    wait
done
