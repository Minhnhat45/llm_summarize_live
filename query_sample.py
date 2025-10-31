import json
import requests
import sys

# ==== FIXED PARAMETERS ====
MODEL = "qwen3-8b-5k-quant-8-2:latest"
URL = "http://157.10.188.151:11434/v1/chat/completions"
TASK = "title"              # "title" or "lead"
STYLE = "đời sống"

LEAD = (
    "Khác các chaebol Hàn Quốc chuộng hàng hiệu hay 'phú nhị đại' Trung Quốc nổi tiếng "
    "với siêu xe, giới tỷ phú Nhật thường chọn sống kín tiếng."
)

CONTENT = """Năm 2018, bộ phim Crazy Rich Asians gây chú ý toàn cầu khi khắc họa giới nhà giàu châu Á. 
Hình ảnh người châu Á từ biểu tượng của sự chăm chỉ đã chuyển thành tầng lớp sẵn sàng chi tiêu 
cho các món xa xỉ như túi xách, trang sức bản giới hạn.
... (rút gọn cho ngắn, giữ nội dung gốc nếu cần)
"""

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
    elif task == "lead":
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
    else:
        return ("Bạn là trợ lý ảo hỗ trợ viết bài báo chí chuyên nghiệp. Hãy trả lời câu hỏi của người dùng.")

def build_user_prompt(task: str, lead: str, content: str) -> str:
    if task == "title":
        return (
            "Hãy viết MỘT tiêu đề duy nhất dựa trên LEAD và CONTENT sau.\n\n"
            f"LEAD:\n{lead}\n\n"
            f"CONTENT:\n{content}\n"
        )
    elif task == "lead":
        return (
            "Tạo LEAD với độ dài từ 1 đến 3 câu cho bài viết dựa trên CONTENT sau.\n\n"
            f"CONTENT:\n{content}\n"
        )
    else:
        return "Xin chào, bạn khỏe không?"

def call_ollama():
    system_prompt = build_system_prompt(TASK, STYLE)
    user_prompt = build_user_prompt(TASK, LEAD, CONTENT)

    # === Log like the bash script ===
    print("\n=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt)
    print(f"=== 🔹 TASK: {TASK} | STYLE: {STYLE} ===")
    print("Sending request to Ollama...")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }

    try:
        response = requests.post(URL, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Handle OpenAI-compatible schema
        if "choices" in data and len(data["choices"]) > 0:
            text = data["choices"][0]["message"]["content"]
            print("\n=== RESPONSE ===")
            print(text.strip())
        else:
            print("Unexpected response format:\n", json.dumps(data, ensure_ascii=False, indent=2))

    except requests.RequestException as e:
        print("❌ Request failed:", e, file=sys.stderr)
        if e.response is not None:
            print("Response:", e.response.text, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    call_ollama()