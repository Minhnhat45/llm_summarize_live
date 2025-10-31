#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import time

# === FIXED PARAMETERS ===
INPUT_CSV = "/mnt/data/test_articles.csv"
OUTPUT_CSV = "/mnt/data/out_articles_result.csv"

ENDPOINT = "http://157.10.188.151:11434/v1/chat/completions"
MODEL = "qwen3-8b-5k-quant-8-2:latest"
TEMPERATURE = 0.7
TOP_P = 0.9

# === PROMPT HELPERS ===
def build_system_prompt(style: str, task: str) -> str:
    if task == "lead":
        return f"[STYLE={style}][TASK=lead]\nBạn là tổng biên tập báo chí dày dạn kinh nghiệm.\nNhiệm vụ: viết LEAD ngắn, rõ, hấp dẫn dựa trên CONTENT.\nChỉ in ra LEAD, không thêm giải thích."
    if task == "title":
        return f"[STYLE={style}][TASK=title]\nBạn là tổng biên tập báo chí dày dạn kinh nghiệm.\nNhiệm vụ: viết TIÊU ĐỀ ngắn gọn, hấp dẫn dựa trên LEAD và CONTENT.\nChỉ in ra TIÊU ĐỀ, không thêm giải thích."
    raise ValueError(f"Unknown task {task}")

def user_prompt_lead(content: str) -> str:
    return f"Hãy viết MỘT LEAD duy nhất dựa trên CONTENT sau:\n\nCONTENT:\n{content}"

def user_prompt_title(lead: str, content: str) -> str:
    return f"Hãy viết MỘT TIÊU ĐỀ duy nhất dựa trên LEAD và CONTENT sau:\n\nLEAD:\n{lead}\n\nCONTENT:\n{content}"

# === REQUEST FUNCTION ===
def chat_request(task: str, style: str, content: str, lead: str = None):
    system_prompt = build_system_prompt(style, task)
    if task == "lead":
        user_prompt = user_prompt_lead(content)
    else:
        user_prompt = user_prompt_title(lead, content)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }

    for i in range(3):
        try:
            r = requests.post(ENDPOINT, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Retry {i+1}/3 due to: {e}")
            time.sleep(2)
    return ""

# === MAIN PROCESS ===
def main():
    df = pd.read_csv(INPUT_CSV)
    if "type" not in df.columns:
        df["type"] = "đời sống"

    df["output_lead"] = ""
    df["output_title"] = ""

    for i, row in df.iterrows():
        content = str(row["content"])
        style = str(row["type"])
        print(f"\n🟢 Processing row {i+1}/{len(df)} | style={style}")

        # Generate lead
        lead = chat_request("lead", style, content)
        df.at[i, "output_lead"] = lead
        print(" → Lead:", lead)

        # Generate title
        title = chat_request("title", style, content, lead)
        df.at[i, "output_title"] = title
        print(" → Title:", title)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
