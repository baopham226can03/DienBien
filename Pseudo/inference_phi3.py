import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import re
import time
import json

# === CONFIG ===
MODEL_NAME = "microsoft/phi-3-medium-4k-instruct"
INPUT_FILE = "unlabel.csv"   # file chứa dữ liệu input (cột "text")
# INPUT_FILE = "a.csv"   # file chứa dữ liệu input (cột "text")
OUTPUT_FILE = "du_pseudo_phi3.csv"
MAX_NEW_TOKENS = 10
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MODE = True  # bật log chi tiết


# === PROMPT (biased toward label=1) ===
def build_prompt(text: str) -> str:
    return f"""<|system|>
You are a JSON generator that outputs only valid JSON objects.
You must classify the following text as *factual* or *non-factual*.

Your priority:
- Be strict when detecting factual statements.


Rules:
- Output ONLY one JSON line.
- The JSON format MUST be exactly: {{"label": 1}} or {{"label": 0}}.
- Do NOT explain or add any other text.
- Respond strictly as JSON, nothing else.

Meaning:
- 1 = clearly factual, verifiable, or objective
- 0 = unfactual, opinion-based, speculative, or subjective

Guidance examples:
- “The sun rises in the east.” → {{"label": 1}}
- “I think this movie is amazing.” → {{"label": 0}}
- “Paris is the capital of France.” → {{"label": 1}}
- “Maybe it will rain tomorrow.” → {{"label": 0}}
- “Temperatures might increase in summer.” → {{"label": 0}}
- “In my opinion, this policy is fair.” → {{"label": 0}}
- “Water boils at 100°C at sea level.” → {{"label": 1}}



<|end|>
<|user|>
Text: "{text}"
<|end|>
<|assistant|>
"""






# === MAIN ===
def main():
    print("Loading model and tokenizer...", flush=True)
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

    print(f"Processing {len(texts)} samples...\n", flush=True)
    start = time.time()

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_prompts = [build_prompt(t) for t in batch_texts]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

        with torch.no_grad():
            batch_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,   # cực thấp để deterministic
                top_p=0.9,
                do_sample=False,   # tắt sampling ngẫu nhiên
                use_cache=True
            )

        for idx, output in enumerate(batch_outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            answer = decoded.split("<|assistant|>")[-1].strip()

            if DEBUG_MODE:
                print("\n" + "-" * 80)
                print(f"📝 Input text: {batch_texts[idx][:200]}...")
                print(f"🔍 Raw model output:\n{answer}")

            # === PARSE JSON (chỉ lấy JSON cuối cùng) ===
            try:
                json_matches = re.findall(r'\{.*?\}', answer)
                if json_matches:
                    json_str = json_matches[-1]  # chỉ lấy JSON cuối cùng
                    result = json.loads(json_str)
                    label = result.get("label", 0)
                    if label not in [0, 1]:
                        label = 0
                else:
                    label = 0
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Parse error: {str(e)}")
                label = 0

            outputs.append(label)

            if DEBUG_MODE:
                print(f"✅ Final label: {label}")
                print("-" * 80)

        del inputs, batch_outputs
        torch.cuda.empty_cache()

    # === SAVE OUTPUT ===
    df["pseudo_label"] = outputs
    df.to_csv(OUTPUT_FILE, index=False)

    end = time.time()
    duration = end - start
    real_count = outputs.count(1)
    fake_count = outputs.count(0)

    print(f"\nProcessing completed in {duration:.2f} seconds")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Statistics: {real_count} REAL, {fake_count} FAKE")
    print(f"\n{'='*60}")
    print(f"✅ Done {len(outputs)} samples in {duration:.1f}s")
    print(f"📊 Label distribution (1=REAL, 0=FAKE):")
    print(f"   - REAL (1): {real_count} ({real_count/len(outputs)*100:.1f}%)")
    print(f"   - FAKE (0): {fake_count} ({fake_count/len(outputs)*100:.1f}%)")
    print(f"📂 Output saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
