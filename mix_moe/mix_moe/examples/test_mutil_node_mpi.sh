cd /workspace/mix_moe

source /root/.bashrc

source activate

export LD_LIBRARY_PATH=/workspace/local/nvshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/workspace/local/gdr_copy/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

ip=$(hostname -I)
echo "本机 IP 地址为: $ip"
echo $OMPI_COMM_WORLD_LOCAL_RANK
echo $OMPI_COMM_WORLD_RANK
echo $OMPI_COMM_WORLD_SIZE

export NCCL_SOCKET_IFNAME=ibs3f0
export NCCL_IB_DISABLE=0

for i in  '4' 
do
     python3 -m mix_moe.examples.mix_moe_ddp  \
     --batch_size=16 --dtype=float16 --num_local_experts $i --name mix_moe
done

# export FMOE_FASTER_SHADOW_ENABLE=0
# export FMOE_FASTER_SCHEDULE_ENABLE=0
# for i in '4' 
# do
#      python3  -m mix_moe.examples.faster_moe_ddp  \
#      --batch_size=16 --dtype=float16 --num_local_experts $i --name fast_moe
# done

# export FMOE_FASTER_SHADOW_ENABLE=1
# export FMOE_FASTER_SCHEDULE_ENABLE=1
# for i in '4' 
# do
#     python3 -m mix_moe.examples.faster_moe_ddp\
#     --batch_size=16 --dtype=float16 --num_local_experts $i --name faster_moe
# done