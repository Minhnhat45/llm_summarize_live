import pandas as pd
from bs4 import BeautifulSoup
df = pd.read_excel("./doi song.xlsx")

sample_content = df["content"][0]
sample_title = df["title"][0]
sample_lead = df["lead"][0]
import pdb
# print(sample_title)
# print("--------------------")
# print(sample_lead)
# print("--------------------")
# print(sample_content)

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
                {
                    "role": "user",
                    "content": (
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
                {
                    "role": "user",
                    "content": (
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

def multitask_instruction_prompt(lead, content):


    prompt_title = """
    [STYLE=VNNews][TASK=title]
    Bạn là tổng biên tập báo chí. Nhiệm vụ: viết MỘT tiêu đề duy nhất theo đúng phong cách tòa soạn.

    QUY TẮC PHONG CÁCH
    - Giọng văn: trung tính, chính xác, chủ động; tránh giật tít.
    - Độ dài: ưu tiên nhỏ hơn 12 từ, tối đa 14 từ.
    - Cú pháp: câu danh đề; không dấu chấm cuối; không dấu “!”; hạn chế “:”.
    - Nội dung: ưu tiên nêu rõ Who/What/Where/When nếu có.
    - Trình bày: viết hoa chuẩn tiếng Việt; không emoji/hashtag; không số liệu suy đoán.

    ĐẦU RA
    - Chỉ in ra tiêu đề duy nhất, không kèm giải thích hay ký tự thừa.
    """.strip()

    prompt_lead = """
    [STYLE=VNNews][TASK=lead]
    Bạn là biên tập viên. Nhiệm vụ: viết LEAD 1–3 câu (khoảng 35–60 từ) theo phong cách tòa soạn.

    QUY TẮC PHONG CÁCH
    - Giọng văn: trung tính, súc tích, ưu tiên câu chủ động.
    - Bao quát: Who/What/Where/When (+ Why/How nếu có dữ kiện).
    - Tính chính xác: không bịa đặt; không suy đoán; số liệu phải khớp.
    - Hình thức: không emoji/hashtag; tránh lặp từ; không trích dẫn dài.

    ĐẦU RA
    - Chỉ in ra LEAD (đoạn văn), không kèm giải thích.
    """.strip()

    sharegpt_payload = {
        "title": {
            "messages": [
                {"role": "system", "content": prompt_title},
                {
                    "role": "user",
                    "content": (
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
                {
                    "role": "user",
                    "content": (
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

if __name__ == "__main__":
    sharegpt_sample = lead_title_prompt(lead=sample_lead, content=clean_content(sample_content))
    print(sharegpt_sample["title"])