import os

os.environ["HF_HOME"] = "/projects/bhuang/.cache/huggingface"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from vllm import SamplingParams, LLM
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.utils import random_uuid

# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# engine_args = AsyncEngineArgs(
llm = LLM(
    model=model_name,
    dtype=torch.float16,
    # tensor_parallel_size=GPU_CONFIG.count,
    gpu_memory_utilization=0.90,
    enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
    # disable_log_stats=True,  # disable logging so we can stream tokens
    # disable_log_requests=True,
)
template = "<s> [INST] {user} [/INST] "

# this can take some time!
# engine = AsyncLLMEngine.from_engine_args(engine_args)
# engine = LLMEngine.from_engine_args(engine_args)


sampling_params = SamplingParams(
    temperature=0.75,
    max_tokens=128,
    repetition_penalty=1.1,
)

user_question = "Lequel est plus lourd entre 1 kilogramme de fer et 1 kilogramme de coton ?"

request_id = random_uuid()
outputs = llm.generate(
    template.format(user=user_question),
    sampling_params,
    # request_id,
)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")