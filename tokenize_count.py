import os
import sys
from transformers import AutoTokenizer
import json
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_tokenizer(model_id: str):
    # trust_remote_code=True for maximum compatibility with Qwen tokenizers.
    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

def count_tokens(tokenizer, text: str, include_special: bool) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=include_special)
    return len(token_ids)

with open("./result_task_2/fo_t2_fewshot_1509_qwen2.5-14b-instruct_0309_final.json", "r", encoding="utf8") as f:
    data = json.load(f)

system_prompt_list = []
input_prompt_list = []
input_prompt_token_length = []
sample = data[0]
list_sample = sample["input"]["system"].split("\n\nVí dụ ")
print(list_sample[0] + "\n")
for sample in list_sample[1:]:
    print("Ví dụ " + str(count_tokens(load_tokenizer(DEFAULT_MODEL), sample, True)) + "\n")

# for item in data:
#     system_prompt_list.append(item["input"]["system"])
#     input_prompt_list.append(item["input"]["user"])

# for s_prompt, u_prompt in zip(system_prompt_list, input_prompt_list):
#     input_prompt_token_length.append(count_tokens(load_tokenizer(DEFAULT_MODEL), u_prompt, True))

# print(input_prompt_token_length)