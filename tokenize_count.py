from __future__ import annotations

import argparse
import json

from typing import Dict, Iterable, Iterator, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_DATA_PATH = "./ecommerce_alpaca_pretty.json"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"


with open(DEFAULT_DATA_PATH, "r", encoding="utf8") as f:
    data = json.load(f)



def count_qwen_tokens(context: str, model_id: str = DEFAULT_QWEN_MODEL, add_special_tokens: bool = False) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokens = len(tokenizer.encode(context, add_special_tokens=add_special_tokens))
    return tokens


if __name__ == "__main__":
    token_length_list = []
    for item in tqdm(data[:10]):
        token_length_list.append(count_qwen_tokens(item["instruction"]))
    print(f"Average token length: {sum(token_length_list) / len(token_length_list)}")
    print(f"Max token length: {max(token_length_list)}")
    print(f"Min token length: {min(token_length_list)}")