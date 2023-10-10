# make LLAMA_CUBLAS=1

CUDA_VISIBLE_DEVICES=0 ./llama.cpp/main \
    -m /work/ckpts/4-bit/Llama-2-70B-chat-GGUF/llama-2-70b-chat.Q4_K_S.gguf \
    --n-gpu-layers 83 \
    --file ./prompt_4096.txt \
    --ctx-size 4096 \
    --ignore-eos \
    -b 4096 \
    -n 64

CUDA_VISIBLE_DEVICES=0 ./llama.cpp/parallel \
    -m /work/ckpts/4-bit/Llama-2-70B-chat-GGUF/llama-2-70b-chat.Q4_K_S.gguf \
    -t 4 -ngl 100 -c 4096 -b 512 -s 1 -np 8 -ns 16 -n 100 -cb