import os
import time
import torch
import argparse
from tqdm import tqdm
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq import exllama_set_max_input_length

from transformers import AutoTokenizer, GenerationConfig


# conda create --name autogptq python=3.10 -y
# conda activate autogptq
# conda install pytorch=2.0.1 torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
# pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/


# CUDA_VISIBLE_DEVICES=0 python run_autogptq.py
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("--long", action="store_true", default=False, help="Using long prompt")
parser.add_argument("--very-long", action="store_true", default=False, help="Using long prompt")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")

args = parser.parse_args()

print("##### Start Testing ##### \n", args)
print("##### CUDA_VISIBLE_DEVICES: {} #####".format(os.environ.get('CUDA_VISIBLE_DEVICES', "0,1,2,3,4,5,6,7")))
num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0,1,2,3,4,5,6,7").split(","))
device = torch.device(f"cuda:{args.local_rank}")

model_name_or_path = "../../ckpts/4-bit/Llama-2-70B-chat-GPTQ"

batch_size = args.batch_size
max_new_tokens = 64
short_sentence = "Hello, my llama is cute,"
long_sentence = """
Hello, my llama is cute, but don't be fooled by its innocent appearance.
This fluffy bundle of fur is full of surprises.
Its name is Larry, and Larry is not your average llama.
Larry has a mischievous twinkle in his eye that hints at the adventures we've had together.
One sunny morning, Larry and I decided to go on an exciting journey through the rolling hills of our hometown.
As we strolled along, Larry's curiosity got the best of him.
He suddenly darted off the path and into a dense thicket of bushes.
I followed, my heart racing, wondering what had caught his attention.
To my amazement, I found Larry standing in front of a hidden treasure chest buried in the underbrush.
With his hoof, he started digging it out, and to our astonishment,
we discovered a trove of ancient maps and mysterious artifacts.
It seemed as though Larry had a sixth sense for adventure, and he had led us to this incredible discovery.
Over the following weeks, Larry and I became the town's talk,
and we began our journey to uncover the secrets of the maps.
We embarked on a series of daring expeditions, solving riddles,
decoding ancient scripts, and traversing treacherous terrain.
Larry's clever instincts and my determination made us an unstoppable team.
As we continued our adventures, Larry's cuteness proved to be a valuable asset.
People couldn't resist his adorable face, which often helped us win the trust
and assistance of locals in our quest. Larry's charm was our secret weapon,
and together we forged lasting friendships with the communities we encountered along the way.
Our adventures with Larry the cute llama took us to the farthest corners of the earth,
unlocking ancient mysteries and uncovering hidden treasures.
But no matter where our journey led, one thing was for certain:
Larry's cuteness was matched only by his sense of adventure,
making him the perfect companion for a life filled with excitement and discovery.
One day, while on an expedition deep in the heart of the Amazon rain,
"""
tokenizer = AutoTokenizer.from_pretrained("../../ckpts/Llama-2-70B-fp16")

if args.long:
    prompt = [long_sentence]
elif args.very_long:
    prompt = [long_sentence * 4]
else:
    prompt = [short_sentence]
prompt = prompt * batch_size

inputs = tokenizer(prompt, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]
print("Prompt length: {}".format(prompt_length))

inputs = inputs.to(device)
quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path)
model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            device="cuda:0",
            low_cpu_mem_usage=True,
            use_triton=False,
            inject_fused_attention=False,
            inject_fused_mlp=True,
            use_cuda_fp16=True,
            quantize_config=quantize_config,
            # model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=False,
            warmup_triton=False,
            disable_exllama=False
        )

model = exllama_set_max_input_length(model, (2048 if args.very_long else 512) * batch_size)

generation_config = GenerationConfig(
        do_sample=False,
        min_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens
    )

perf1 = []
perf2 = []
for _ in tqdm(range(10)):
    torch.cuda.synchronize()
    start = time.perf_counter()
    generated_ids = model.generate(input_ids=inputs.input_ids, generation_config=generation_config)[0]
    torch.cuda.synchronize()
    end = time.perf_counter()
    latency = end - start
    # latency_per_token = latency / (generated_ids.shape[1] * batch_size)
    latency_per_token = latency / max_new_tokens
    perf1.append(latency)
    perf2.append(latency_per_token)
perf1 = perf1[3:]
perf2 = perf2[3:]
e2e_latency = sum(perf1) / len(perf1) * 1000
print("Average E2E latency (ms): ", e2e_latency)
print("Average E2E throughput (sample/s): ", batch_size / (sum(perf1) / len(perf1)))
e2e_token_latency = sum(perf2) / len(perf2) * 1000
print("Average E2E token latency: (ms/token)", e2e_token_latency)
print("Average E2E token throughput (token/s): ", 1000 * batch_size / e2e_token_latency)

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("Outputs: ", outputs)