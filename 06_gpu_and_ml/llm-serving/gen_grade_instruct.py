import os
import time
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from file_utils import thread_safe_jsonl_dump

def run_generation(gen_func):
    input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/grade_prompt_b.txt"
    # input_prompt_file = "/home/bhuang/nlp/vigogne/data/generation/grade_prompt_c.txt"

    input_file = "/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8.jsonl"
    output_file = "/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedmixtral8x22b.jsonl"
    instruction_field = "evolved_instruction"

    # input_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed.jsonl"
    # output_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed_promptevaluatedmixtral8x22b.jsonl"
    # instruction_field = "translated_instruction"

    # input_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed.jsonl"
    # output_file = "/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed_promptevaluatedmixtral8x7b.jsonl"
    # instruction_field = "translated_instruction"

    output_field = "output"
    max_parallel_requests = 32

    with open(input_prompt_file) as f:
        prompt_template = f.read()

    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # debug
    # dataset = dataset.select(range(20))

    if os.path.exists(output_file):
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(instruction_field)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=instruction_field, num_proc=4)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    def process_item(item):
        # grade prompt
        instruction = prompt_template.format(question=item[instruction_field])
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
