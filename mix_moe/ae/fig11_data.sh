export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --local-dir-use-symlinks False  --resume-download gpt2 --local-dir /root/models/gpt2
huggingface-cli download --local-dir-use-symlinks False  --resume-download google-bert/bert-base-uncased --local-dir /root/models/bert

export LD_LIBRARY_PATH=/root/local/nvshmem/lib:$LD_LIBRARY_PATH

export FMOE_FASTER_SHADOW_ENABLE=1
export FMOE_FASTER_SCHEDULE_ENABLE=1
export CUDA_LAUNCH_BLOCKING=1

rm /root/mix_moe/ae/output_data/e2e_model.csv

echo "name,MOE-GPT,MOE-BERT,MOE-Transformer-xl" >> ./root/mix_moe/ae/output_data/e2e_model.csv

cd ~/mix_moe/mix_moe/examples/models


# bert
if [ "$1" == "bert" ];then
    mpirun --allow-run-as-root -np 4 \
        -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
        -x MASTER_PORT='7777' -x LOCAL_SIZE=1      \
        -x PATH -bind-to none -host localhost      \
        -map-by slot -mca pml ob1 -mca btl ^openib  \
        python3 -m tutel.launcher.run -m moe_bert_ddp    \
        --model_dim 768 --hidden_size 3072  --num_tokens 512 \
        --moe  --multi_gpu  --batch_size=32 --dtype=float16 \
        --num_local_experts 16 >> /root/mix_moe/ae/output_data/fig11_data.txt 
fi
# cat /root/mix_moe/ae/output_data/fig11_data.txt | grep "Summary" > /root/mix_moe/ae/output_data/fig11_data.txt

# gpt
if [ "$1" == "gpt2" ];then
    rm /root/mix_moe/ae/output_data/fig11_data.txt 
    mpirun --allow-run-as-root -np 4 \
        -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
        -x MASTER_PORT='7777' -x LOCAL_SIZE=1      \
        -x PATH -bind-to none -host localhost      \
        -map-by slot -mca pml ob1 -mca btl ^openib  \
        python3 -m tutel.launcher.run -m moe_gpt_ddp    \
        --model_dim 768 --hidden_size 3072  --num_tokens 512 \
        --multi_gpu  --batch_size=8 --dtype=float16 \
        --num_local_experts 16  >> /root/mix_moe/ae/output_data/fig11_data.txt 
fi



if [ "$1" == "transformer-xl" ];then
    cd /root/mix_moe/mix_moe/examples/transformer-xl


    mpirun --allow-run-as-root -np 4  \
       -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
       -x MASTER_PORT='7777' -x LOCAL_SIZE=1  \
       -x PATH -bind-to none -host localhost  \
       -map-by slot -mca pml ob1 -mca btl ^openib \
       python3 -m tutel.launcher.run -m train_ddp  --fp16 \
       --dtype=float16      --cuda     --data ./data/enwik8/ \
       --dataset enwik8     --n_layer 12     --d_model 512     \
       --n_head 8     --d_head 64     --d_inner 2048     --dropout 0.1 \
       --dropatt 0.0     --optim adam     --lr 0.00025     --warmup_step 0  \
       --max_step 400000     --tgt_len 512     --mem_len 512     --eval_tgt_len 128 \
       --batch_size 16     --multi_gpu     --gpu0_bsz -1   --moe-num-expert 16 --moe-top-k 2 \
        --name naive --cuda >> /root/mix_moe/ae/output_data/fig11_data.txt 

    mpirun --allow-run-as-root -np 4  \
       -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
       -x MASTER_PORT='7777' -x LOCAL_SIZE=1  \
       -x PATH -bind-to none -host localhost  \
       -map-by slot -mca pml ob1 -mca btl ^openib \
       python3 -m tutel.launcher.run -m train_ddp  --fp16 \
       --dtype=float16      --cuda     --data ./data/enwik8/ \
       --dataset enwik8     --n_layer 12     --d_model 512     \
       --n_head 8     --d_head 64     --d_inner 2048     --dropout 0.1 \
       --dropatt 0.0     --optim adam     --lr 0.00025     --warmup_step 0  \
       --max_step 400000     --tgt_len 512     --mem_len 512     --eval_tgt_len 128 \
       --batch_size 16     --multi_gpu     --gpu0_bsz -1   --moe-num-expert 16 --moe-top-k 2 \
        --name fastermoe --moe --cuda >> /root/mix_moe/ae/output_data/fig11_data.txt 

    mpirun --allow-run-as-root -np 4  \
       -H 127.0.0.1:4 -x MASTER_ADDR='127.0.0.1' \
       -x MASTER_PORT='7777' -x LOCAL_SIZE=1  \
       -x PATH -bind-to none -host localhost  \
       -map-by slot -mca pml ob1 -mca btl ^openib \
       python3 -m tutel.launcher.run -m train_ddp  --fp16 \
       --dtype=float16      --cuda     --data ./data/enwik8/ \
       --dataset enwik8     --n_layer 12     --d_model 512     \
       --n_head 8     --d_head 64     --d_inner 2048     --dropout 0.1 \
       --dropatt 0.0     --optim adam     --lr 0.00025     --warmup_step 0  \
       --max_step 400000     --tgt_len 512     --mem_len 512     --eval_tgt_len 128 \
       --batch_size 16     --multi_gpu     --gpu0_bsz -1   --moe-num-expert 16 --moe-top-k 2 \
        --name mix_moe --moe --cuda >> /root/mix_moe/ae/output_data/fig11_data.txt 
fi