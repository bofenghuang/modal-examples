# vllm offline inference
#
#
# # Fast inference with vLLM (Mixtral 8x7B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# We are running the [Mixtral 8x7B Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model here,
# which is a mixture-of-experts model finetuned for conversation.
# You can expect ~3 minute cold starts.
# For a single request, the throughput is over 50 tokens/second.
# The larger the batch of prompts, the higher the throughput (up to hundreds of tokens per second).
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
import time

import modal

MODEL_DIR = "/model"

# https://modal.com/pricing
GPU_CONFIG = modal.gpu.A100(memory=80, count=2)
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_REVISION = "1e637f2d7cb0a9d6fb1922f305cb784995190a83"

# GPU_CONFIG = modal.gpu.A100(memory=80, count=4)
# MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"
# MODEL_REVISION = "95d063951382d47385fe7b36e202b68639e5c066"

# GPU_CONFIG = modal.gpu.A100(memory=80, count=4)
# MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
# MODEL_REVISION = "e8cf5276ae3e97cfde8a058e64a636f2cde47820"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# For this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# like Mixtral 8x7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.
#
# Mixtral is beefy, at nearly 100 GB in `safetensors` format, so this can take some time -- at least a few minutes.
def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin", "*.gguf"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We’ll start from a Dockerhub image recommended by `vLLM`, and use
# run_function to run the function defined above to ensure the weights of
# the model are saved within the container image.

image = (
    # todo: install flash-attn
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "vllm==0.4.1",  # LLM serving
            "huggingface_hub==0.22.2",  # download models from the Hugging Face Hub
            "hf-transfer==0.1.6",  # download models faster
        ]
        # "vllm==0.4.0.post1",
        # "torch==2.1.2",
        # "transformers==4.39.3",
        # "ray==2.10.0",
        # "hf-transfer==0.1.6",
        # "huggingface_hub==0.22.2",
    )
    # modal.Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.3.3")
    # # modal.Image.from_registry("ghcr.io/huggingface/text-generation-inference:2.0.2")
    # .dockerfile_commands("ENTRYPOINT []")
    # .pip_install(
    #     "vllm==0.4.0.post1",
    #     "torch==2.1.2",
    #     "transformers==4.39.3",
    #     "ray==2.10.0",
    #     "hf-transfer==0.1.6",
    #     "huggingface_hub==0.22.2",
    # )
    # modal.Image.from_registry("vllm/vllm-openai:v0.4.2", add_python="3.10")
    # modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    # .pip_install(
    #     "vllm==0.4.0.post1",
    #     "torch==2.1.2",
    #     "transformers==4.39.3",
    #     "ray==2.10.0",
    #     "hf-transfer==0.1.6",
    #     "huggingface_hub==0.22.2",
    #     "flash-attn==2.5.5"
    # )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App("example-vllm-inference", image=image)  # Note: prior to April 2024, "app" was called "stub"

# Using `image.imports` allows us to have a reference to vLLM in global scope without getting an error when our script executes locally.
with image.imports():
    import vllm


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions),
# using a `load_model` method decorated with `@modal.enter`. This enables us to load the model into memory just once,
# every time a container starts up, and to keep it cached on the GPU for subsequent invocations of the function.
#
# The `vLLM` library allows the code to remain quite clean.
@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 24,
    container_idle_timeout=60 * 5,
    # allow_concurrent_inputs=10,
)
class Model:
    @modal.enter()
    def load_model(self):

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        # Tip: models that are not fully implemented by Hugging Face may require `trust_remote_code=true`.
        self.llm = vllm.LLM(
            MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            # gpu_memory_utilization=1.0,
        )
        # mistral
        self.template = "[INST] {user} [/INST]"
        # llama-3
#         self.template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

# {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# """

        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    def generate(self, user_questions):
        prompts = [self.template.format(user=q) for q in user_questions]

        sampling_params = vllm.SamplingParams(
            # temperature=0.75,
            # top_p=1,
            # max_tokens=256,
            # presence_penalty=1.15,
            # -- generate instruction
            # temperature=0.7,
            # max_tokens=4096,
            # repetition_penalty=1.1,
            # -- grade prompt
            max_tokens=1024,
            # llama-3
            # stop=["<|eot_id|>"],
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            # print(
            #     f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
            #     f"\n{COLOR['BLUE']}{output.outputs[0].text}",
            #     "\n\n",
            #     sep=COLOR["ENDC"],
            # )
            # time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return [output.outputs[0].text for output in result]


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@app.local_entrypoint()
def main():
    """
    questions = [
        # # Coding questions
        # "Implement a Python function to compute the Fibonacci numbers.",
        # "Write a Rust function that performs binary exponentiation.",
        # "How do I allocate memory in C?",
        # "What are the differences between Javascript and Python?",
        # "How do I find invalid indices in Postgres?",
        # "How can you implement a LRU (Least Recently Used) cache in Python?",
        # "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        # "Can you explain how a decision tree algorithm works in machine learning?",
        # "How would you design a simple key-value store database from scratch?",
        # "How do you handle deadlock situations in concurrent programming?",
        # "What is the logic behind the A* search algorithm, and where is it used?",
        # "How can you design an efficient autocomplete system?",
        # "What approach would you take to design a secure session management system in a web application?",
        # "How would you handle collision in a hash table?",
        # "How can you implement a load balancer for a distributed system?",
        # # Literature
        # "What is the fable involving a fox and grapes?",
        # "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        # "Who does Harry turn into a balloon?",
        # "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        # "Describe a day in the life of a secret agent who's also a full-time parent.",
        # "Create a story about a detective who can communicate with animals.",
        # "What is the most unusual thing about living in a city floating in the clouds?",
        # "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        # "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        # "Tell a story about a musician who discovers that their music has magical powers.",
        # "In a world where people age backwards, describe the life of a 5-year-old man.",
        # "Create a tale about a painter whose artwork comes to life every night.",
        # "What happens when a poet's verses start to predict future events?",
        # "Imagine a world where books can talk. How does a librarian handle them?",
        # "Tell a story about an astronaut who discovered a planet populated by plants.",
        # "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        # "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # # History
        # "What were the major contributing factors to the fall of the Roman Empire?",
        # "How did the invention of the printing press revolutionize European society?",
        # "What are the effects of quantitative easing?",
        # "How did the Greek philosophers influence economic thought in the ancient world?",
        # "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        # "How did decolonization in the 20th century change the geopolitical map?",
        # "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # # Thoughtfulness
        # "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        # "In a dystopian future where water is the most valuable commodity, how would society function?",
        # "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        # "What could be the potential implications of contact with an advanced alien civilization?",
        # "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        # # Math
        # "What is the product of 9 and 8?",
        # "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        # "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        # "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        # "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        # "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # # Facts
        # "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        # "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        # "What was Project A119 and what were its objectives?",
        # "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        # "What is the 'Emu War' that took place in Australia in the 1930s?",
        # "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        # "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        # "What are 'zombie stars' in the context of astronomy?",
        # "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        # "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        # "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        # "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    ]
    model = Model()
    model.generate.remote(questions)
    # quit()
    """

    import os

    # import time
    # from tqdm import tqdm
    from datasets import load_dataset

    # from concurrent.futures import ThreadPoolExecutor, as_completed
    # from file_utils import thread_safe_jsonl_dump

    input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/grade_prompt_b.txt"
    # input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/grade_prompt_c.txt"

    input_file = "/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8.jsonl"
    # output_file = "/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedmixtral8x22b.jsonl"
    output_file = "/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b.jsonl"
    instruction_field = "evolved_instruction"

    # input_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed.jsonl"
    # # output_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedmixtral8x22b.jsonl"
    # output_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedllama370b.jsonl"
    # instruction_field = "translated_instruction"

    # input_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed.jsonl"
    # # output_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedmixtral8x7b.jsonl"
    # output_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedllama370b.jsonl"
    # instruction_field = "translated_instruction"

    output_field = "output"

    with open(input_prompt_file) as f:
        prompt_template = f.read()

    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # debug
    # dataset = dataset.select(range(20))
    # dataset = dataset.select(range(4000))

    # if os.path.exists(output_file):
    #     existing_dataset = load_dataset("json", data_files=output_file, split="train")
    #     existing_values = existing_dataset.unique(instruction_field)
    #     existing_values = set(existing_values)
    #     print(f"Found {len(existing_values):,d} existing examples in {output_file}")

    #     dataset = dataset.filter(lambda x: x not in existing_values, input_columns=instruction_field, num_proc=4)
    #     print(f"Filtered to {dataset.num_rows:,d} examples")

    start_time = time.perf_counter()

    # grade prompt
    dataset = dataset.map(
        lambda example: {"eval_prompt": prompt_template.format(question=example[instruction_field])}, num_proc=8
    )

    # init model
    model = Model()

    # infer
    result = model.generate.remote(dataset["eval_prompt"])
    dataset = dataset.map(lambda _, idx: {output_field: result[idx]}, with_indices=True, num_proc=8)

    # def process_function(batch):
    #     batch["eval_prompt"] = [prompt_template.format(question=inst) for inst in batch[instruction_field]]
    #     batch[output_field] = model.generate.remote(batch["eval_prompt"])
    #     jsonl_dump(batch, output_file, mode="a", default=str, ensure_ascii=False)
    #     return batch

    # infer
    # dataset = dataset.map(
    #     lambda example: {output_field: model.generate.remote(example["eval_prompt"])},
    #     lambda example: process_function,
    #     batched=True,
    #     batch_size=1024 * 4,
    #     num_proc=1,
    #     drop_last_batch=False,
    #     desc="Generating..",
    # )

    dataset.to_json(output_file, orient="records", lines=True, force_ascii=False, mode="a")

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file}"
    )
