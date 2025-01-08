cd /root/mix_moe

source /root/.bashrc

export LD_LIBRARY_PATH=/root/local/nvshmem/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

for i in '4' '8' '16' '32' '64'
do
    mpirun --allow-run-as-root -np 4 \
     -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
     -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
     -x PATH -bind-to none -host localhost \
     -map-by slot -mca pml ob1 -mca btl ^openib \
     python3 -m tutel.launcher.run -m mix_moe.examples.mix_moe_ddp  \
     --batch_size=16 --dtype=float16 --num_local_experts $i --name mix_moe
done


export FMOE_FASTER_SHADOW_ENABLE=0
export FMOE_FASTER_SCHEDULE_ENABLE=0
for i in '4' '8' '16' '32' '64'
do
    mpirun --allow-run-as-root -np 4 \
     -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
     -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
     -x PATH -bind-to none  \
     -map-by slot -mca pml ob1 -mca btl ^openib \
     python3 -m tutel.launcher.run -m mix_moe.examples.faster_moe_ddp  \
     --batch_size=16 --dtype=float16 --num_local_experts $i --name fast_moe
done

export FMOE_FASTER_SHADOW_ENABLE=1
export FMOE_FASTER_SCHEDULE_ENABLE=1

for i in '4' '8' '16' '32' '64'
do
    mpirun --allow-run-as-root -np 4 \
     -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
     -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
     -x PATH -bind-to none \
     -map-by slot -mca pml ob1 -mca btl ^openib \
     python3 -m tutel.launcher.run -m mix_moe.examples.faster_moe_ddp  \
     --batch_size=16 --dtype=float16 --num_local_experts $i --name faster_moe
done

# /workspace/local/openmpi/bin/mpirun --allow-run-as-root -np 8 \
#  -H 120.1.1.1:4,120.1.1.101:4 -x MASTER_ADDR='120.1.1.1' \
#  -x MASTER_PORT='29501' -x LOCAL_SIZE=1 \
#  -x NVSHMEM_REMOTE_TRANSPORT='ucx' \
#  -x PATH -bind-to none  \
#  -map-by slot -mca pml ob1 \
#  -mca btl_openib_allow_ib true \
#  --mca plm_rsh_args "-p 8888" \
#  /workspace/nvshmem_test_mutli_node/build/test

# /workspace/local/openmpi/bin/mpirun --allow-run-as-root -np 4 \
#  -H 120.1.1.101:4 -x MASTER_ADDR='120.1.1.101' \
#  -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
#  -x PATH -bind-to none  \
#  -map-by slot -mca pml ob1 \
#  -mca btl_openib_allow_ib true \
#  --mca plm_rsh_args "-p 8888" \
#  bash /workspace/mix_moe/mix_moe/examples/test_mutil_node_mpi.sh

#  python3 -m tutel.launcher.run -m mix_moe.examples.mix_moe_ddp  \
#  --batch_size=16 --dtype=float16 --num_local_experts 16 --name mix_moe

#  mpirun --allow-run-as-root -np 4 \
#      -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
#      -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
#      -x PATH -bind-to none -host localhost \
#      -map-by slot -mca pml ob1 -mca btl ^openib \
#      python3 -m tutel.launcher.run -m train_ddp  \
#      --fp16 --dtype=float16 \
#      --cuda \
#     --data ./data/enwik8/ \
#     --dataset enwik8 \
#     --n_layer 12 \
#     --d_model 512 \
#     --n_head 8 \
#     --d_head 64 \
#     --d_inner 2048 \
#     --dropout 0.1 \
#     --dropatt 0.0 \
#     --optim adam \
#     --lr 0.00025 \
#     --warmup_step 0 \
#     --max_step 400000 \
#     --tgt_len 512 \
#     --mem_len 512 \
#     --eval_tgt_len 128 \
#     --batch_size 32 \
#     --multi_gpu \
#     --gpu0_bsz -1 \
#     --moe --moe-num-expert 16 --moe-top-k 2 \
#     --work_dir ./run_enwik8_base_moe