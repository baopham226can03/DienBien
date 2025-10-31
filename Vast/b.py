import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-3-medium-4k-instruct"

# Kiểm tra CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tải tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # float16 giúp tiết kiệm VRAM
    device_map="auto"           # tự động phân bố model lên GPU
)

# Prompt thử nghiệm
prompt = """
Decide if the following news statement is REAL or FAKE.
Return only one number:
1 for REAL
0 for FAKE
Only return 1 digit, no explain (No "response" or any character, only and only one digit).

News:
{"Statement: Wisconsin is on pace to double the number of layoffs this year. Speaker: katrina-shankland (democrat party) Topic: jobs Analysis: She cited layoff notices received by the state. But those arent actual layoffs. In the time frame she cited the states added about 30,300 jobs."}
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Sinh đầu ra
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.0,
        top_p=0.9,
        do_sample=False
    )

# Giải mã kết quả
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
