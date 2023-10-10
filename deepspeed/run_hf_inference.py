import os
import time
import torch
import deepspeed
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("-o", "--optimum", action="store_true", default=False, help="see https://huggingface.co/docs/optimum/bettertransformer/overview")
parser.add_argument("-d", "--deepspeed", action="store_true", default=False, help="using deepspeed")
parser.add_argument("--data-type", type=str, default="fp16", help="Data type, [fp16, int8, int4, nf4]")
parser.add_argument("--long", action="store_true", default=False, help="Using long prompt")
parser.add_argument("--model", type=str, default="13b", help="default is 13b")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")

args = parser.parse_args()

print("##### Start Testing ##### \n", args)
print("##### CUDA_VISIBLE_DEVICES: {} #####".format(os.environ.get('CUDA_VISIBLE_DEVICES', "0,1,2,3,4,5,6,7")))
num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0,1,2,3,4,5,6,7").split(","))
device = torch.device(f"cuda:{args.local_rank}")

# model_name = "Qwen/Qwen-7B-Chat"  # pip install einops tiktoken transformers_stream_generator
if args.model == "70b":
    model_name = "../ckpts/Llama-2-70B-fp16"
elif args.model == "180b":
    model_name = "../ckpts/falcon-180B-chat-synthetic"
else:
    model_name = "../ckpts/Llama-2-13B-fp16"

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
tokenizer = AutoTokenizer.from_pretrained("../ckpts/Llama-2-13B-fp16")

prompt = [short_sentence if not args.long else long_sentence] * batch_size
inputs = tokenizer(prompt, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]
print("Prompt length: {}".format(prompt_length))

if args.deepspeed:
    print("Using Deepspeed")
    # deepspeed --no_local_rank --include localhost:6,7 run_hf_inference.py -b 1 -d
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().half()
    model = deepspeed.init_inference(
        model,
        mp_size=num_gpus,
        dtype=torch.bfloat16 if args.data_type != "int8" else torch.int8,
        replace_with_kernel_inject=True,
        max_out_tokens=prompt_length + max_new_tokens
    )
    if args.data_type == "int8":
        for layer in model.module.model.layers:
            if isinstance(layer, DeepSpeedGPTInference):
                setattr(layer, 'dtype', torch.int8)
else:
    inputs = inputs.to(device)
    if args.data_type == "nf4":
        print("Using nf4")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config).eval()
    elif args.data_type == "int8":
        from torch.ao.quantization import (
            get_default_qconfig_mapping,
            get_default_qat_qconfig_mapping,
            QConfigMapping,
        )
        import torch.ao.quantization.quantize_fx as quantize_fx

        model_to_quantize = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()

        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
        example_inputs = inputs.input_ids
        model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        model = quantize_fx.convert_fx(model_prepared)
    else:
        # device_map="auto"这一行等同于.cuda(), 即使后面再.half()也无法再释放显存
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()
        print("Convert to float16")
        # model = model.half()
        # model = model.cuda()
    if args.optimum:
        print("Using Huggingface's optimum")
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)

# calculate output time
generate_kwargs = dict(
    max_new_tokens=max_new_tokens,
    do_sample=False,
    repetition_penalty=1.0,
    use_cache=True
)

perf1 = []
perf2 = []
for _ in range(10):
    torch.cuda.synchronize()
    start = time.perf_counter()
    if args.deepspeed:
        inputs = inputs.to(device)
        model.cuda().to(device)
    generated_ids = model.generate(inputs.input_ids, **generate_kwargs)
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