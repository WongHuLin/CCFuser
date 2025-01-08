export LD_LIBRARY_PATH=/root/local/nvshmem/lib:$LD_LIBRARY_PATH

export FMOE_FASTER_SHADOW_ENABLE=1
export FMOE_FASTER_SCHEDULE_ENABLE=1
export CUDA_LAUNCH_BLOCKING=1

export LOCALITY=1
rm ./output_data/kernel*
echo "kernel, locality, time, all2all" > ./output_data/kernel_eval_nvlink.csv
mpirun --allow-run-as-root -np 4\
    -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1'  \
    -x MASTER_PORT='7777' \
    python /root/mix_moe/mix_moe/kernel_evaluation.py >> ./output_data/kernel_eval_nvlink.txt
sed -n "6,45p" ./output_data/kernel_eval_nvlink.txt >> ./output_data/kernel_eval_nvlink.csv


export LOCALITY=0
echo "kernel, locality, time, all2all, num_expert" > ./output_data/kernel_eval_expert.csv
mpirun --allow-run-as-root -np 4\
    -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1'  \
    -x MASTER_PORT='7777' \
    python /root/mix_moe/mix_moe/kernel_evaluation.py >> ./output_data/kernel_eval_expert.txt

sed -n "6,29p" ./output_data/kernel_eval_expert.txt >> ./output_data/kernel_eval_expert.csv
