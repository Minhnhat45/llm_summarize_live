import json

fixed_out = "train_data_09_10_2025_fixed.jsonl"
with open("train_data_09_10_2025_norm.jsonl", "r", encoding="utf-8") as f_in, \
     open(fixed_out, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        # line itself is a string containing escaped JSON -> parse once
        obj = json.loads(line)       # this turns it into a string again
        if isinstance(obj, str):
            obj = json.loads(obj)    # parse second time to get dict
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Fixed dataset written to {fixed_out}")

