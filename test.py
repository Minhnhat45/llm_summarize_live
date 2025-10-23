
from typing import Dict, List
from datasets import load_dataset


def build_chat(sample: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Build chat messages list from ShareGPT-style JSONL:
    {"conversations": [{"role": "system", "content": ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Args:
        sample: one JSON record
        remove_think: if True, strip <think>...</think> sections

    Returns:
        List[Dict[role, content]]
    """
    messages = []
    conv = sample.get("conversations", [])
    for msg in conv:
        role = msg.get("role", "").strip()
        content = (msg.get("content") or "").strip()

        # Skip empty content
        if not content:
            continue

        messages.append({"role": role, "content": content})

    # Sanity fallback: if dataset missing system role, add default
    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    return messages

raw = load_dataset("json", data_files={"train": "./train_data_16_10_2025_fixed.jsonl"})
ds_train = raw["train"]

print(ds_train[1])
msgs = build_chat(ds_train[1])
for m in msgs:
    print(m)