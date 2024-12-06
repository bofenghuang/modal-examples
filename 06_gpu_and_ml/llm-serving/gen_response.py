import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from file_utils import thread_safe_jsonl_dump
from tqdm import tqdm


def run_generation(gen_func):
    input_file = "/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b_5rp.jsonl"
    output_file = "/projects/bhuang/corpus/text/llm/generated/wild_chat_1m/wild_chat_1m_french_processed_continuedmixtral8x22b_5rpr.jsonl"
    id_field = "instruction"
    message_field = "messages"
    output_field = "output"
    max_parallel_requests = 32

    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # debug
    # dataset = dataset.select(range(20))
    # dataset = dataset.select(range(1))

    if os.path.exists(output_file):
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(id_field)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=id_field, num_proc=4)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    def process_item(item):
        # item[output_field] = Model().generate.remote(instruction)
        item[output_field] = gen_func(item[message_field])
        thread_safe_jsonl_dump(item, output_file, mode="a")

    start_time = time.perf_counter()

    translated_data = []
    with tqdm(total=dataset.num_rows, desc="Genrating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
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
