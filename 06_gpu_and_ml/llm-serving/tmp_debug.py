import time
import random
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from file_utils import thread_safe_jsonl_dump

input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/gen_instruct/general.txt"
input_file = "/projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered.jsonl"
# output_file = "/projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered_generatedinstructions.jsonl"
max_parallel_requests = 1

min_examples = 5
max_examples = 10
min_generated_examples = 3
num_iterations = 2_000

with open(input_prompt_file) as f:
    prompt_template = f.read()

df = pd.read_json(input_file, lines=True)
topics = df.groupby("topic")["instruction"].apply(list).to_dict()
print(f"Loaded {len(topics.keys())} topics")

def process_item():
    topic_name, instructions = random.choice(list(topics.items()))
    num_examples = random.choice(range(min_examples, max_examples + 1))
    num_examples = min(num_examples, len(instructions))
    sampled_instructions = random.sample(instructions, num_examples)

    num_generated_examples = random.choice(range(min_generated_examples, num_examples + 1))
    context = "\n\n".join([f"Example {idx + 1} :\n{inst}" for idx, inst in enumerate(sampled_instructions)])
    prompt = prompt_template.format(context=context, batch_size=num_generated_examples)

    item = {"topic": topic_name}

    print(prompt)

    # item["output"] = Model().generate.remote(prompt)

    # thread_safe_jsonl_dump(item, output_file, mode="a")

process_item()

# start_time = time.perf_counter()


# translated_data = []
# with tqdm(total=num_iterations, desc="Genrating") as pbar:
#     with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
#         futures = {executor.submit(process_item): _ for _ in range(num_iterations)}
#         for future in as_completed(futures):
#             translated_data.append(future.result())
#             pbar.update(1)

# print(
#     f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
#     f" data is saved in {output_file}"
# )