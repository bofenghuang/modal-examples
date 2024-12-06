import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from file_utils import thread_safe_jsonl_dump
from tqdm import tqdm


def run_generation(gen_func):
    # input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/multi_turn_continuation_b.txt"
    input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/multi_turn_continuation_c.txt"

    # input_file = "/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed.jsonl"
    # output_file = "/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b.jsonl"

    input_file = "/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b_4rprp.jsonl"
    output_file = "/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b_5r.jsonl"

    id_field = "instruction"
    message_field = "messages"
    output_field = "output"
    num_final_criterias = 2
    max_parallel_requests = 32

    with open(input_prompt_file) as f:
        prompt_template = f.read()

    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # debug
    # dataset = dataset.select(range(20))
    # dataset = dataset.shuffle(10)
    # dataset = dataset.select(range(10_000))

    if os.path.exists(output_file):
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(id_field)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=id_field, num_proc=4)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    def process_item(item, prompt_template):
        m = re.search(r"<methods>\n(.+)\n</methods>", prompt_template, flags=re.DOTALL).groups()[0]
        criterias = m.split("\n")
        # criteria_names = [c.split(": ", 1)[0] for c in criterias]
        # criteria_idx = random.choice(range(len(criteria_names)))
        # prompt_template = re.sub(m, criterias[criteria_idx], prompt_template)

        formatted_criteria = "\n".join(random.sample(criterias, num_final_criterias))
        prompt_template = re.sub(m, formatted_criteria, prompt_template)

        # format prompt
        formatted_conversation = "\n\n".join([f'{turn["role"].capitalize()} : {turn["content"]}' for turn in item[message_field]])
        instruction = prompt_template.format(conversation=formatted_conversation)
        # item["eval_prompt"] = instruction
        # item[output_field] = Model().generate.remote(instruction)
        item[output_field] = gen_func(instruction)
        thread_safe_jsonl_dump(item, output_file, mode="a")

    start_time = time.perf_counter()

    translated_data = []
    with tqdm(total=dataset.num_rows, desc="Genrating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    prompt_template,
                ): item
                for item in dataset
            }
            for future in as_completed(futures):
                translated_data.append(future.result())
                pbar.update(1)

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file}"
    )
