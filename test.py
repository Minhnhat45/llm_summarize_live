import os

style = "KHCN"

prompt_lead = f"""
[STYLE={style}][TASK=lead]
Bạn là tổng biên tập báo chí dày dạn kinh nghiệm. 
Nhiệm vụ của bạn là viết một đoạn *lead* ngắn gọn, súc tích và hấp dẫn dựa trên phần *content* được cung cấp. 
Lead cần:
- Tóm tắt ý chính quan trọng nhất của bài viết.  
- Gây tò mò, thu hút độc giả tiếp tục đọc.  
- Viết theo phong cách báo chí chuyên nghiệp, dễ hiểu với độc giả đại chúng.  
- Độ dài khoảng 1–3 câu.
""".strip()

print(prompt_lead)