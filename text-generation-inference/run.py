import os
import time
import asyncio
import argparse
from huggingface_hub import InferenceClient, AsyncInferenceClient

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("--long", action="store_true", default=False, help="Using long prompt")
parser.add_argument("--output-length", type=int, default=64, help="Max output sequence length")

args = parser.parse_args()
print(args)

prompt_10 = "Hello, my llama is cute,"
prompt_512 = """Hello, my llama is cute, but don't be fooled by its innocent appearance.
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
One day, while on an expedition deep in the heart of the Amazon rain,"""
prompt_2048 = prompt_512 * 4

prompt = prompt_2048 if args.long else prompt_512

# client = InferenceClient(model="http://127.0.0.1:8080")
# warm up
# for _ in range(5):
#     output = client.text_generation(prompt=prompt, max_new_tokens=64)

# Batched inference
# https://huggingface.co/docs/huggingface_hub/main/en/guides/inference#async-client
# https://github.com/huggingface/text-generation-inference/issues/355
client = AsyncInferenceClient(model="http://127.0.0.1:8080")
perf = []
for _ in range(5):
    jobs = [client.text_generation(prompt, max_new_tokens=args.output_length) for _ in range(args.batch_size)]
    start = time.time()
    async def batch():
        return await asyncio.gather(*jobs)
    results = asyncio.run(batch())
    end = time.time()
    batch_latency = end - start
    perf.append(batch_latency)
perf = perf[3:]
latency = sum(perf) / len(perf)
latency_per_token = 1000 * latency / args.output_length
print("E2E latency: {:.2f} ms".format(latency * 1000))
print("Output token throughput: {:.2f} token/s".format(1000 / latency_per_token * args.batch_size))
print("Output token latency: {:.2f} ms".format(latency_per_token))
print("##########################################################")