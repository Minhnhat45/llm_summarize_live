#!/bin/bash

MODEL="qwen3-8b-5k-quant-8-2:latest"
URL="http://157.10.188.151:11434/v1/chat/completions"

# TASK can be: title | lead
TASK="title"

# STYLE can be: đời sống | du lịch | khoa học công nghệ
STYLE="đời sống"

# Sample

LEAD="Khác các chaebol Hàn Quốc chuộng hàng hiệu hay 'phú nhị đại' Trung Quốc nổi tiếng với siêu xe, giới tỷ phú Nhật thường chọn sống kín tiếng."
CONTENT="Năm 2018, bộ phim Crazy Rich Asians gây chú ý toàn cầu khi khắc họa giới nhà giàu châu Á. Hình ảnh người châu Á từ biểu tượng của sự chăm chỉ đã chuyển thành tầng lớp sẵn sàng chi tiêu cho các món xa xỉ như túi xách, trang sức bản giới hạn.\nỞ Singapore, biểu tượng \"người siêu giàu\" không còn xa vời. Cứ 30 người dân thì có một người là triệu phú, giúp quốc gia này xếp thứ sáu châu Á về số cá nhân có tài sản ròng cao (HNWI). HNWI là người sở hữu ít nhất một triệu USD tài sản có thể đầu tư, không tính bất động sản hay hàng tiêu dùng.\nTrung Quốc đứng thứ hai, với nhiều cá nhân góp mặt trong top 100 người giàu nhất thế giới theo Forbes. Thế hệ thừa kế nước này thường gây chú ý trên mạng xã hội với siêu xe và thử thách #fallingstarschallenge, nơi họ \"ngã\" khỏi xe Aston Martin giữa đống hàng hiệu.\nDẫn đầu châu Á về số lượng HNWI là Nhật Bản, quốc gia xếp thứ hai thế giới chỉ sau Mỹ. Cứ 17 hộ gia đình Nhật thì có một hộ sở hữu hơn một triệu USD. Trung bình, mỗi HNWI Nhật nắm giữ 2,5 triệu USD, tổng cộng 7,7 nghìn tỷ USD năm 2017, vượt 6,5 nghìn tỷ USD của Trung Quốc.\nTuy vậy, Crazy Rich Asians không chọn Nhật Bản làm bối cảnh, bởi giới giàu nước này kín tiếng, sống và tiêu dùng giản dị, khó phân biệt với tầng lớp trung lưu. Truyền thông gọi họ là \"người giàu kín đáo\" của châu Á.\nCó nhiều lý do khiến giới giàu Nhật Bản không khoe tài sản. Đầu tiên, truyền thống và văn hóa Nhật Bản chịu ảnh hưởng sâu sắc từ Thiền tông, coi khiêm tốn là một đức tính. Tinh thần này gắn với triết lý wa (hài hòa) và kenkyo (khiêm tốn), thể hiện qua câu tục ngữ \"cái đinh nhô lên sẽ bị đóng xuống\".\nTừ thời Edo (1603–1868), các quy định nghiêm ngặt đã cấm thường dân phô trương tài sản để duy trì trật tự và tránh ghen tị. Truyền thống đó kéo dài đến nay, hình thành chuẩn mực xã hội coi việc khoe của là thiếu tinh tế.\nTinh thần khiêm tốn này thể hiện rõ trong giới thừa kế hiện đại. Con cái của Masayoshi Son, nhà sáng lập SoftBank, và Tadashi Yanai, CEO Uniqlo, hiếm khi xuất hiện trên mạng xã hội hay khoe tài sản. Họ thường học trường công, làm việc bình thường hoặc sống kín tiếng ở nước ngoài để giữ gìn hình ảnh gia đình. Theo các nhà phân tích văn hóa trên Quora năm 2019, điều này phản ánh giá trị Nhật Bản coi trọng sự hài hòa tập thể hơn phô trương cá nhân.\nĐồng thời, người Nhật xem giàu có là trách nhiệm hơn đặc quyền cá nhân. Truyền thống này bắt nguồn từ lịch sử hình thành tầng lớp thượng lưu Nhật Bản. Các zaibatsu như Mitsui hay Mitsubishi ra đời từ thời Minh Trị (1868–1912), phát triển qua nhiều thế hệ, tạo nên khái niệm \"tiền cũ\" - tài sản đi cùng danh dự và trách nhiệm duy trì uy tín gia tộc.\nTrong văn hóa ấy, sự kín đáo không chỉ là phép lịch sự mà còn là biểu hiện của đẳng cấp thật sự. Báo cáo năm 2024 của hãng CarterJMRN cho biết giới siêu giàu Nhật (HNWI) hiện sở hữu tổng tài sản 7,7 nghìn tỷ USD, cao nhất châu Á, nhưng hiếm khi xuất hiện công khai.\nSự ổn định xã hội cũng góp phần định hình thái độ này. Với hệ số Gini chỉ 0,33 – thấp hơn nhiều nước phát triển, Nhật Bản duy trì mức chênh lệch giàu nghèo tương đối nhỏ. Khi khoảng cách xã hội không quá rõ rệt, nhu cầu thể hiện địa vị qua vật chất cũng giảm đi.\nBên cạnh đó, ảnh hưởng của Thần đạo và Phật giáo khiến người Nhật coi trọng \"giàu có tinh thần\" hơn của cải vật chất. Trái lại, ở một số quốc gia châu Á khác, sự giàu có đôi khi được sử dụng như công cụ khẳng định vị thế xã hội, dẫn tới những tranh luận về khoảng cách giàu nghèo, vấn đề từng được phản ánh trong phim Parasite của Hàn Quốc.\nHiện, thế hệ rich kid Nhật Bản vẫn giữ lối sống kín đáo dù chịu ảnh hưởng từ mạng xã hội và chủ nghĩa cá nhân phương Tây. Trên Instagram hay TikTok, họ có thể chia sẻ cuộc sống riêng nhưng hiếm khi phô trương sự xa hoa.\nNếu phú nhị đại (thế hệ thừa kế thứ hai) của Trung Quốc tìm kiếm sự công nhận và chaebol Hàn Quốc dùng tài sản để củng cố ảnh hưởng, giới nhà giàu Nhật lại thể hiện \"sự tự tin im lặng\" coi địa vị là cách sống, không phải tài sản."
TITLE="Tại sao nhà giàu Nhật không khoe của?"


# System Prompt

SYSTEM_PROMPT() {
  local task=$1
  local style=$2
  if [[ $task == "title" ]]; then
    cat <<EOF
[STYLE=${style}][TASK=${task}]
Bạn là tổng biên tập báo chí dày dạn kinh nghiệm.
Nhiệm vụ của bạn là viết một tiêu đề ngắn gọn, hấp dẫn, đúng phong cách báo chí chuyên nghiệp và dễ hiểu với độc giả đại chúng, dựa trên phần *lead* và *content* được cung cấp.
Title cần:
- Chỉ in ra tiêu đề, KHÔNG kèm giải thích.
- Ngắn gọn dưới 15 từ, dễ hiểu, rõ ràng.
- Không sử dụng '?', '!', ';', '"'
EOF
  else
    cat <<EOF
[STYLE=${style}][TASK=${task}]
Bạn là tổng biên tập báo chí dày dạn kinh nghiệm.
Nhiệm vụ của bạn là viết một đoạn *lead* ngắn gọn, súc tích và hấp dẫn dựa trên phần *content* được cung cấp.
Lead cần:
- Tóm tắt ý chính quan trọng nhất của bài viết.
- Gây tò mò, thu hút độc giả tiếp tục đọc.
- Viết theo phong cách báo chí chuyên nghiệp, dễ hiểu với độc giả đại chúng.
- Độ dài khoảng từ 1 đến 3 câu.
EOF
  fi
}



# User Prompt
USER_PROMPT() {
  local task=$1
  if [[ $task == "title" ]]; then
    cat <<EOF
Hãy viết MỘT tiêu đề duy nhất dựa trên LEAD và CONTENT sau.

LEAD:
$LEAD

CONTENT:
$CONTENT
EOF
  else
    cat <<EOF
Tạo LEAD với độ dài từ 1 đến 3 câu cho bài viết dựa trên CONTENT sau.

CONTENT:
$CONTENT
EOF
  fi
}


# Request Model
SYSTEM_CONTENT=$(SYSTEM_PROMPT "$TASK" "$STYLE")
USER_CONTENT=$(USER_PROMPT "$TASK")

# Log prompt
echo -e "\n\n=== SYSTEM PROMPT ==="
echo "$SYSTEM_CONTENT"

echo -e "\n\n=== USER PROMPT ==="
echo "$USER_CONTENT"


# Normalize JSON format
SYSTEM_ESCAPED=$(echo "$SYSTEM_CONTENT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
USER_ESCAPED=$(echo "$USER_CONTENT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

echo "=== 🔹 TASK: $TASK | STYLE: $STYLE ==="
echo "Sending request to Ollama..."


curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": ${SYSTEM_ESCAPED}},
      {\"role\": \"user\", \"content\": ${USER_ESCAPED}}
    ],
    \"temperature\": 0.6,
    \"top_p\": 0.9,
    \"max_tokens\": 4096
  }"