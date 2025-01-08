export LD_LIBRARY_PATH=/workspace/local/nvshmem/lib:$LD_LIBRARY_PATH
 mpirun --allow-run-as-root -np 4 \
     -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
     -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
     -x PATH -bind-to none -host localhost \
     -map-by slot -mca pml ob1 -mca btl ^openib \
     python3 -m tutel.launcher.run -m train_ddp  \
     --fp16 --dtype=float16 \
     --cuda \
    --data ./data/enwik8/ \
    --dataset enwik8 \
    --n_layer 12 \
    --d_model 512 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 2048 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 400000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 32 \
    --multi_gpu \
    --gpu0_bsz -1 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --work_dir ./run_enwik8_base_moe \
    --name mix_moe

export FMOE_FASTER_SHADOW_ENABLE=1
export FMOE_FASTER_SCHEDULE_ENABLE=1

 mpirun --allow-run-as-root -np 4 \
     -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
     -x MASTER_PORT='7777' -x LOCAL_SIZE=1 \
     -x PATH -bind-to none -host localhost \
     -map-by slot -mca pml ob1 -mca btl ^openib \
     python3 -m tutel.launcher.run -m train_ddp  \
     --fp16 --dtype=float16 \
     --cuda \
    --data ./data/enwik8/ \
    --dataset enwik8 \
    --n_layer 12 \
    --d_model 512 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 2048 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 400000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 32 \
    --multi_gpu \
    --gpu0_bsz -1 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --work_dir ./run_enwik8_base_moe \
    --name faster_moe