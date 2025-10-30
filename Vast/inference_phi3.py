import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import time

MODEL_NAME = "microsoft/phi-3-medium-4k-instruct"
INPUT_FILE = "du_unlabeled.csv"   # cột: text
OUTPUT_FILE = "du_pseudo_phi3.csv"
MAX_NEW_TOKENS = 10
BATCH_SIZE = 1  # decoder-only -> sinh lần lượt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_prompt(text):
    return f"""
News Content: {text}.
Based on the content of the news article provided, determine whether the news is Fake or Real.
Just select one of these two options. No explanation required. Only return 0 for Real news and 1 for Fake news (1 character only).
    """

def main():
    print(f"🧠 Loading model: {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    df = pd.read_csv(INPUT_FILE)
    texts = df["text"].astype(str).tolist()
    outputs = []

    print(f"🚀 Running inference for {len(texts)} samples ...")
    start = time.time()

    for text in tqdm(texts):
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        # lấy phần sau cùng (assistant output)
        answer = decoded.split("<|assistant|>")[-1].strip().lower()
        label = 1 if "giả" in answer or "fake" in answer else 0
        outputs.append(label)

    df["pseudo_label"] = outputs
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done {len(outputs)} samples in {time.time()-start:.1f}s")
    print(f"📂 Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
