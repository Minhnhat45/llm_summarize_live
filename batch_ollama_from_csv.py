#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import sys
import requests
import pandas as pd
from typing import Dict, Any

# ======== FIXED CONFIG ========
MODEL = "qwen3-8b-5k-quant-8-2:latest"
URL = "http://157.10.188.151:11434/v1/chat/completions"
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_TOKENS = 4096
# Backoff between requests (seconds) in case the endpoint needs pacing
SLEEP_BETWEEN_CALLS = 0.2
INPUT_CSV = "/mnt/data/test_articles.csv"          # change if needed
OUTPUT_CSV = "/mnt/data/test_articles_outputs.csv" # change if needed
# ==============================


def build_system_prompt(task: str, style: str) -> str:
    if task == "title":
        return (
            f"[STYLE={style}][TASK={task}]\n"
            "Bạn là tổng biên tập báo chí dày dạn kinh nghiệm.\n"
            "Nhiệm vụ của bạn là viết một tiêu đề ngắn gọn, hấp dẫn, đúng phong cách báo chí chuyên nghiệp "
            "và dễ hiểu với độc giả đại chúng, dựa trên phần *lead* và *content* được cung cấp.\n"
            "Title cần:\n"
            "- Chỉ in ra tiêu đề, KHÔNG kèm giải thích.\n"
            "- Ngắn gọn dưới 15 từ, dễ hiểu, rõ ràng.\n"
            "- Không sử dụng '?', '!', ';', '\"'\n"
        )
    else:
        return (
            f"[STYLE={style}][TASK={task}]\n"
            "Bạn là tổng biên tập báo chí dày dạn kinh nghiệm.\n"
            "Nhiệm vụ của bạn là viết một đoạn *lead* ngắn gọn, súc tích và hấp dẫn dựa trên phần *content* "
            "được cung cấp.\n"
            "Lead cần:\n"
            "- Tóm tắt ý chính quan trọng nhất của bài viết.\n"
            "- Gây tò mò, thu hút độc giả tiếp tục đọc.\n"
            "- Viết theo phong cách báo chí chuyên nghiệp, dễ hiểu với độc giả đại chúng.\n"
            "- Độ dài khoảng từ 1 đến 3 câu.\n"
        )


def build_user_prompt_for_lead(content: str) -> str:
    return (
        "Tạo LEAD với độ dài từ 1 đến 3 câu cho bài viết dựa trên CONTENT sau.\n\n"
        f"CONTENT:\n{content}\n"
    )


def build_user_prompt_for_title(lead: str, content: str) -> str:
    return (
        "Hãy viết MỘT tiêu đề duy nhất dựa trên LEAD và CONTENT sau.\n\n"
        f"LEAD:\n{lead}\n\n"
        f"CONTENT:\n{content}\n"
    )


def post_chat(url: str, model: str, system_prompt: str, user_prompt: str,
              temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_text(resp_json: Dict[str, Any]) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        # Fallback: dump JSON for debugging
        return json.dumps(resp_json, ensure_ascii=False)


def main():
    # Load input CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"❌ Could not read input CSV at {INPUT_CSV}: {e}", file=sys.stderr)
        sys.exit(1)

    # Expect columns: 'type' (STYLE), 'content' (CONTENT)
    # Optional: 'lead' or 'title' exist but we will ignore them and regenerate
    required_cols = ["type", "content"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing required column '{col}' in input CSV.", file=sys.stderr)
            sys.exit(1)

    outputs = []

    for idx, row in df.iterrows():
        style = str(row["type"]) if not pd.isna(row["type"]) else "đời sống"
        content = str(row["content"]) if not pd.isna(row["content"]) else ""

        # 1) Ask for LEAD
        sp_lead = build_system_prompt(task="lead", style=style)
        up_lead = build_user_prompt_for_lead(content=content)

        try:
            resp_lead = post_chat(URL, MODEL, sp_lead, up_lead, TEMPERATURE, TOP_P, MAX_TOKENS)
            output_lead = extract_text(resp_lead).strip()
        except Exception as e:
            output_lead = f"[ERROR generating lead: {e}]"

        time.sleep(SLEEP_BETWEEN_CALLS)

        # 2) Ask for TITLE using the generated lead + content
        sp_title = build_system_prompt(task="title", style=style)
        up_title = build_user_prompt_for_title(lead=output_lead, content=content)

        try:
            resp_title = post_chat(URL, MODEL, sp_title, up_title, TEMPERATURE, TOP_P, MAX_TOKENS)
            output_title = extract_text(resp_title).strip()
        except Exception as e:
            output_title = f"[ERROR generating title: {e}]"

        outputs.append({
            **row.to_dict(),
            "output_lead": output_lead,
            "output_title": output_title,
        })

        # Optional: progress log
        print(f"[{idx+1}/{len(df)}] style={style} | lead_len={len(output_lead)} | title='{output_title[:60]}{'...' if len(output_title)>60 else ''}'")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Build result DataFrame
    out_df = pd.DataFrame(outputs)

    # Save to CSV
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Saved outputs to: {OUTPUT_CSV}")
    return out_df


if __name__ == "__main__":
    main()
