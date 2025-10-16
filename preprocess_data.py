import pandas as pd
from bs4 import BeautifulSoup
import json
import os
import pdb


def clean_content(html_content):
    """
    Clean all the html form from content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "figure", "div"]):
        tag.decompose()
    # pdb.set_trace()
    text = soup.get_text(separator="\n")
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    
    return clean_text

def lead_title_prompt(lead, content):
    """
    Create ShareGPT prompt format.
    """
    prompt_title = """
    Bạn là tổng biên tập báo chí dày dạn kinh nghiệm. 
    Nhiệm vụ của bạn là viết một tiêu đề ngắn gọn, hấp dẫn, đúng phong cách báo chí chuyên nghiệp và dễ hiểu với độc giả đại chúng, dựa trên phần *lead* và *content* được cung cấp.
    """.strip()

    # Prompt sinh lead
    prompt_lead = """
    Bạn là tổng biên tập báo chí dày dạn kinh nghiệm. 
    Nhiệm vụ của bạn là viết một đoạn *lead* ngắn gọn, súc tích và hấp dẫn dựa trên phần *content* được cung cấp. 
    Lead cần:
    - Tóm tắt ý chính quan trọng nhất của bài viết.  
    - Gây tò mò, thu hút độc giả tiếp tục đọc.  
    - Viết theo phong cách báo chí chuyên nghiệp, dễ hiểu với độc giả đại chúng.  
    - Độ dài khoảng 1–3 câu.
    """.strip()
    
    sharegpt_payload = {
        "title": {
            "messages": [
                {"role": "system", "content": prompt_title},
                {"role": "user", "content": (
                        "Hãy viết MỘT tiêu đề duy nhất dựa trên LEAD và CONTENT sau.\n\n"
                        f"LEAD:\n{lead}\n\n"
                        f"CONTENT:\n{content}\n\n"
                        "YÊU CẦU ĐẦU RA:\n"
                        "- Chỉ in ra tiêu đề, KHÔNG kèm giải thích.\n"
                        "- Ngắn gọn, rõ ràng, dễ hiểu; tránh giật tít phản cảm.\n"
                        "- Ưu tiên < 14 từ nếu có thể."
                    ),
                },
                {"role": "assistant", "content": ""}
            ]
        },
        "lead": {
            "messages": [
                {"role": "system", "content": prompt_lead},
                {"role": "user", "content": (
                        "Tạo LEAD 1–3 câu cho bài viết dựa trên CONTENT sau.\n\n"
                        f"CONTENT:\n{content}\n\n"
                        "YÊU CẦU ĐẦU RA:\n"
                        "- Súc tích, hấp dẫn, tóm tắt ý chính quan trọng nhất.\n"
                        "- Gợi mở, khuyến khích độc giả đọc tiếp.\n"
                        "- Không dùng emoji/hashtag; không bịa số liệu."
                    ),
                },
                {"role": "assistant", "content": ""}
            ]
        }
    }
    
    return sharegpt_payload

def multitask_instruction_prompt(lead, content, title, style):

    
    prompt_title = f"""
    [STYLE={style}][TASK=title]
    Bạn là tổng biên tập báo chí dày dạn kinh nghiệm. 
    Nhiệm vụ của bạn là viết một tiêu đề ngắn gọn, hấp dẫn, đúng phong cách báo chí chuyên nghiệp và dễ hiểu với độc giả đại chúng, dựa trên phần *lead* và *content* được cung cấp.
    Title cần:
    - Chỉ in ra tiêu đề, KHÔNG kèm giải thích.
    - Ngắn gọn dưới 15 từ, dễ hiểu, rõ ràng.
    - Không sử dụng '?', '!', ';', '"'
    """.strip()

    prompt_lead = f"""
    [STYLE={style}][TASK=lead]
    Bạn là tổng biên tập báo chí dày dạn kinh nghiệm. 
    Nhiệm vụ của bạn là viết một đoạn *lead* ngắn gọn, súc tích và hấp dẫn dựa trên phần *content* được cung cấp. 
    Lead cần:
    - Tóm tắt ý chính quan trọng nhất của bài viết.  
    - Gây tò mò, thu hút độc giả tiếp tục đọc.  
    - Viết theo phong cách báo chí chuyên nghiệp, dễ hiểu với độc giả đại chúng.  
    - Độ dài khoảng từ 1 đến 3 câu.
    """.strip()

    sharegpt_payload = {
        "title": {
            "conversations": [
                {"role": "system", "content": prompt_title},
                {"role": "user", "content": (
                        "Hãy viết MỘT tiêu đề duy nhất dựa trên LEAD và CONTENT sau.\n\n"
                        f"LEAD:\n{lead}\n\n"
                        f"CONTENT:\n{content}\n\n"
                    ),
                },
                {"role": "assistant", "content": f"{title}"}
            ]
        },
        "lead": {
            "conversations": [
                {"role": "system", "content": prompt_lead},
                {"role": "user", "content": (
                        "Tạo LEAD với độ dài từ 1 đến 3 câu cho bài viết dựa trên CONTENT sau.\n\n"
                        f"CONTENT:\n{content}\n\n"
                    ),
                },
                {"role": "assistant", "content": f"{lead}"}
            ]
        }
    }
    
    return sharegpt_payload

def general_instruction_prompt(instruction, output):
    """
    General task sẽ bao gồm các prompt liên quan đến xử lý đoạn văn để sát với yêu cầu LLM là một biên tập báo chí hơn.
    Sẽ bao gồm nhiều task xử lý đoạn văn khác nhau như: tóm tắt, viết lại, phân loại, hỏi đáp,...
    """
    prompt_other = """
    [STYLE=general][TASK=general]
    Bạn là tổng biên tập báo chí dày dạn kinh nghiệm. Nhiệm vụ của bạn là tham gia trò chuyện đa mục đích (general chat) với người dùng, bao gồm:
    - Trả lời tự nhiên, ngắn gọn nhưng đầy đủ ý; khi cần có thể diễn giải chi tiết.
    - Sử dụng giọng văn rõ ràng, dễ tiếp cận, tránh dùng từ ngữ gây khó hiểu.
    - Nếu nội dung thuộc phạm vi nhạy cảm, hãy từ chối lịch sự và hướng người dùng sang lựa chọn an toàn, hữu ích hơn.
    Mục tiêu: Giúp cuộc trò chuyện trở nên tự nhiên, hữu ích và hiệu quả nhất cho người dùng.
    """
    sharegpt_payload = {
        "conversations": [
            {"role": "system", "content": prompt_other},
            {"role": "user", "content": (instruction)},
            {"role": "assistant", "content": (output)}
        ]
    }
    return sharegpt_payload
    
    
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def handle_categories(data_path, style):
    df = pd.read_excel(data_path)

    n_ok, n_skip = 0, 0
    for _, row in df.iterrows():
        lead_val = str(row["lead"]) if pd.notna(row["lead"]) else ""
        content_val = str(row["content"]) if pd.notna(row["content"]) else ""
        title_val = str(row["title"]) if pd.notna(row["title"]) else ""

        if not content_val.strip():
            n_skip += 1
            continue

        content_clean = clean_content(content_val)

        payload = multitask_instruction_prompt(
            lead=lead_val.strip(),
            content=content_clean,
            title=title_val.strip(),
            style=style
        )

        # dump the whole multitask payload (title + lead) in one line
        append_jsonl(out_jsonl, payload["title"])
        append_jsonl(out_jsonl, payload["lead"])
        n_ok += 1
    print(f"Done. Wrote {n_ok} samples to {out_jsonl}. Skipped {n_skip} empty content rows.")

if __name__ == "__main__":
    out_jsonl = "./train_data_16_10_2025.jsonl"
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)
    

    # prepare lead/title prompt

    handle_categories("./doi song.xlsx", "đời sống")
    handle_categories("./du lich.xlsx", "du lịch")
    handle_categories("./KHCN.xlsx", "khoa học công nghệ")
    
    
    
    # prepare general instruction prompt
    # with open("./ecommerce_alpaca_pretty.json", "r", encoding="utf8") as f:
    #     ecommerce_data = json.load(f)
    # count_instructions = 0
    # for sample in ecommerce_data:
    #     if "đoạn văn" in sample["instruction"].strip().lower():
    #         sharegpt_sample = general_instruction_prompt(instruction=sample["instruction"], output=sample["output"])
    #         # print(sharegpt_sample["title"]["messages"])
    #         # print(sharegpt_sample["title"]["messages"])
    #         # print("-----"*10)
    #         append_jsonl(out_jsonl, sharegpt_sample)
    #         count_instructions += 1
    # print(f"Total general instructions: {count_instructions}")