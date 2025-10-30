import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import time

MODEL_NAME = "microsoft/phi-3-medium-4k-instruct"
INPUT_FILE = "a.csv"   # cột: text
OUTPUT_FILE = "du_pseudo_phi3.csv"
MAX_NEW_TOKENS = 10
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MODE = True  # Set True to see what model outputs

# --- Prompt builder ---
def build_prompt(text):
    return f"""<|system|>
You are an expert fact-checker and news analyst with deep expertise in identifying misinformation, verifying claims, and detecting fake news patterns. Your task is to analyze news articles with extreme precision using journalistic standards and critical thinking.
<|end|>
<|user|>
Analyze the following news article and determine if it is REAL or FAKE news.

**ARTICLE:**
{text}

**ANALYSIS FRAMEWORK:**
Apply these critical evaluation criteria:
1. **Credibility Signals**: Check for verifiable facts, named sources, specific dates/locations, and expert quotes
2. **Red Flags**: Look for sensationalism, clickbait headlines, emotional manipulation, unverified claims, missing attribution
3. **Logical Consistency**: Assess if claims are coherent, plausible, and supported by evidence
4. **Language Patterns**: Detect bias, propaganda techniques, exaggeration, or misleading framing
5. **Information Quality**: Evaluate specificity, context, and whether key details are present or suspiciously absent

**INSTRUCTIONS:**
- Think step-by-step but keep reasoning concise
- Weigh evidence carefully
- Be decisive based on the strongest indicators
- Output ONLY a single digit:
  * **0** = FAKE news (misleading, unverified, false)
  * **1** = REAL news (credible, verifiable, factual)

**YOUR VERDICT (0 or 1 ONLY):**<|end|>
<|assistant|>"""

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
    raw_responses = []  # Store raw model outputs for debugging

    print(f"🚀 Running inference for {len(texts)} samples in batches of {BATCH_SIZE} ...")
    start = time.time()

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_prompts = [build_prompt(t) for t in batch_texts]

        # tokenize cả batch cùng lúc
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

        with torch.no_grad():
            batch_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )

        # decode từng sample trong batch
        for idx, output in enumerate(batch_outputs):
            decoded = tokenizer.decode(output.cpu(), skip_special_tokens=True)
            # Lấy phần sau <|assistant|>
            answer = decoded.split("<|assistant|>")[-1].strip()
            
            # Debug: in ra 5 sample đầu tiên
            if DEBUG_MODE and len(outputs) < 5:
                print(f"\n{'='*60}")
                print(f"Sample {len(outputs)+1}:")
                print(f"Raw answer: {repr(answer[:100])}")
                print(f"{'='*60}")
            
            # Parse output: tìm số 0 hoặc 1
            if "0" in answer[:10]:  # Check first 10 chars for the digit
                label = 0
            elif "1" in answer[:10]:
                label = 1
            else:
                # Fallback: keyword detection
                answer_lower = answer.lower()
                if "fake" in answer_lower or "giả" in answer_lower or "false" in answer_lower:
                    label = 1
                elif "real" in answer_lower or "thật" in answer_lower or "true" in answer_lower:
                    label = 0
                else:
                    label = 0  # Default to real if uncertain
            
            outputs.append(label)
            raw_responses.append(answer[:50])  # Save first 50 chars

        # Giải phóng GPU memory sau mỗi batch
        del inputs
        del batch_outputs
        torch.cuda.empty_cache()

    df["pseudo_label"] = outputs
    df["model_response"] = raw_responses  # Add raw responses for inspection
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"✅ Done {len(outputs)} samples in {time.time()-start:.1f}s")
    print(f"📊 Label distribution:")
    print(f"   - Label 0 (FAKE): {outputs.count(0)} ({outputs.count(0)/len(outputs)*100:.1f}%)")
    print(f"   - Label 1 (REAL): {outputs.count(1)} ({outputs.count(1)/len(outputs)*100:.1f}%)")
    print(f"📂 Saved: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
