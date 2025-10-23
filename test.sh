curl http://180.148.130.10:11434/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "qwen3-8b-vne:latest",
    "messages": [
      {"role": "user", "content": "Xin chào, bạn khỏe không?"}
    ]
  }'