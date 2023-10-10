
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import argparse, os, math, time
import pandas, fastparquet
import torch
import torch.nn.functional as F
from conversion.tokenize import get_tokens
from conversion.quantize import list_live_tensors

import sys
import json

torch.cuda._lazy_init()
torch.set_printoptions(precision = 10)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

# git clone https://github.com/turboderp/exllamav2
# cd exllamav2
# EXLLAMA_NOCOMPILE= python setup.py install --user
# CUDA_VISIBLE_DEVICES=0 python test_inference.py -m /data2/chaon/repo/LLM/ckpts/4-bit/Llama2-70B-exl2 -p -t 64 --very-long
parser = argparse.ArgumentParser(description = "Test inference on ExLlamaV2 model")
parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = 128, help = "Number of rows to apply from dataset")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-p", "--prompt", action = "store_true", help = "Generate from prompt (basic sampling settings)")
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("--long", action = "store_true", help = "prompt length 512")
parser.add_argument("--very-long", action = "store_true", help = "prompt length 2048")
parser.add_argument("-t", "--tokens", type = int, default = 128, help = "Max no. tokens")
parser.add_argument("-ps", "--prompt_speed", action = "store_true", help = "Test prompt processing (batch) speed over context length")
parser.add_argument("-s", "--speed", action = "store_true", help = "Test raw generation speed over context length")

# Initialize model and tokenizer

model_init.add_args(parser)
args = parser.parse_args()
model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args)

batch_size = args.batch_size
model.config.max_batch_size = batch_size
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

if args.long:
    prompt = [long_sentence]
elif args.very_long:
    prompt = [long_sentence * 8]
else:
    prompt = [short_sentence]
prompt = prompt * batch_size

# Test generation

if args.prompt:

    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        ids = tokenizer.encode(prompt)
        tokens_prompt = ids.shape[-1]

        print(f" -- Warmup...")

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()
        print(f" -- Generating...")
        print()

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        time_begin = time.time()

        output = generator.generate_simple(prompt, settings, args.tokens, token_healing = True)

        torch.cuda.synchronize()
        time_prompt = time.time()

        time_end = time.time()

    print(output)
    print()

    total_gen = time_end - time_begin
    print(f" -- Response generated in {total_gen:.2f} seconds, {args.tokens} tokens, {args.tokens / total_gen:.2f} tokens/second (includes prompt eval.)")

    cache = None


# Test perplexity

if args.eval_dataset:

    with torch.inference_mode():

        eval_dataset = args.eval_dataset
        eval_rows = args.eval_rows
        eval_length = args.eval_length

        print(f" -- Running perplexity test")
        print(f" -- Dataset: {eval_dataset}")
        print(f" -- Tokenizing eval data, {eval_rows} rows x {eval_length} tokens...")

        eval_tokens = get_tokens(eval_rows, eval_length, eval_dataset, tokenizer)

        print(f" -- Inference", end = "")
        sys.stdout.flush()

        logprob_sum = 0.0
        logprob_count = 0

        cache = ExLlamaV2Cache(model, max_seq_len = eval_length) if eval_length > model.config.max_input_len else None

        for i in range(eval_tokens.shape[0]):

            if i % 10 == 0: print(".", end = "")
            sys.stdout.flush()

            input_ids = eval_tokens[i:i+1, :]

            input_ids = input_ids[:, :]
            if cache is not None: cache.current_seq_len = 0
            logits = model.forward(input_ids, cache)
            logits = logits[:, :-1, :]
            logits = logits.float() + 1e-10

            target_ids = input_ids[:, 1:].to(logits.device)

            log_probs = F.log_softmax(logits, dim = -1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum += token_log_probs.sum().item()
            logprob_count += target_ids.numel()

        print()

        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

        print(f" -- Evaluation perplexity: {perplexity:.4f}")


# Test prompt speed

if args.prompt_speed:

    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        ids = torch.randint(0, model.config.vocab_size - 1, (1, model.config.max_seq_len))

        print(f" -- Warmup...")

        model.forward(ids[:, -1:])

        print(f" -- Measuring prompt speed...")

        current_len = 128
        while True:

            time_begin = time.time()

            cache.current_seq_len = 0
            model.forward(ids[:, :current_len], cache, preprocess_only = True)
            torch.cuda.synchronize()

            time_end = time.time()
            tps = current_len / (time_end - time_begin)

            print(f" ** Length {current_len:>5} tokens: {tps:>11.4f} t/s")

            current_len_ = current_len
            current_len = min(current_len + 128, model.config.max_seq_len)
            if current_len == current_len_: break

    cache = None


# Test token speed

if args.speed:

    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        print(f" -- Measuring token speed...")
        ids = tokenizer.encode("X")
        model.forward(ids[:, :])

        current_idx = ids.shape[-1]
        next_stop = 128

        while True:

            time_begin = time.time()

            tokens = next_stop - current_idx
            for i in range(tokens):

                logits = model.forward(ids[:, -1:], cache)
                sample = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
                ids = torch.cat((ids, sample), dim=-1)

            time_end = time.time()
            tps = tokens / (time_end - time_begin)

            print(f" ** Position {current_idx:>5} + {tokens:>3} tokens: {tps:>9.4f} t/s")

            current_idx = next_stop
            next_stop = min(next_stop + 128, model.config.max_seq_len)
            if next_stop == current_idx: break
