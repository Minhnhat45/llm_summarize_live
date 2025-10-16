from __future__ import annotations

import argparse
import json
import multiprocessing as mp

from typing import Dict, Iterable, Iterator, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
import os
DEFAULT_DATA_PATH = "./ecommerce_alpaca_pretty.json"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"

access_token = os.getenv("HF_TOKEN")


_tokenizer: AutoTokenizer | None = None


def _init_tokenizer(model_id: str, hf_token: str) -> None:
    """Load tokenizer once per worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)


def count_qwen_tokens(context: str, model_id: str = DEFAULT_QWEN_MODEL, add_special_tokens: bool = False) -> int:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)

    tokens = len(_tokenizer.encode(context, add_special_tokens=add_special_tokens))
    return tokens

import ast
if __name__ == "__main__":
    # with open(DEFAULT_DATA_PATH, "r", encoding="utf8") as f:
    #     data = json.load(f)
    instructions = []
    with open("./train_data_09_10_2025.jsonl", "r", encoding="utf-8") as f:
        data = f.readlines()
    for item in data:
        item_dict = ast.literal_eval(item)
        all_sample = ""
        for contents in item_dict["conversations"]:
            all_sample += contents["role"] + ": " + contents["content"] + "\n"
        instructions.append(all_sample)
    
    with mp.Pool(processes=14, initializer=_init_tokenizer, initargs=(DEFAULT_QWEN_MODEL, access_token)) as pool:
        token_length_list = list(
            tqdm(pool.imap(count_qwen_tokens, instructions), total=len(instructions))
        )
    # print(token_length_list)
    avg_token_length = int(sum(token_length_list) / len(token_length_list))
    print(f"Average token length: {avg_token_length}")
    print(f"Max token length: {max(token_length_list)}")
    print(f"Min token length: {min(token_length_list)}")
    with open("./train_data_09_10_2025_norm.jsonl", "w", encoding="utf-8") as f:
        for sample, length in zip(data, token_length_list):
            if length in range(200, 4097):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            

