import json

with open("./ecommerce_alpaca.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("./ecommerce_alpaca_pretty.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)