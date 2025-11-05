import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import re
import time

# === CONFIG ===
MODEL_NAME = "microsoft/phi-3-medium-4k-instruct"
INPUT_FILE = "train_unlabeled.csv"   # file chứa 1000 mẫu
# INPUT_FILE = "a.csv"   # file chứa 1000 mẫu
OUTPUT_FILE = "du_pseudo_phi3.csv"
MAX_NEW_TOKENS = 5     # sinh ngắn, chỉ cần số 0/1
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MODE = False  # tắt debug để giảm logging



# === PROMPT ===
def build_prompt(text: str) -> str:
    return f"""<|system|>
You are an expert fact-checker. Carefully read the following news article and make an objective, balanced, and fair judgment.
Your task is to decide whether the article is primarily:
1 = TRUE (accurate, verifiable)
0 = FALSE (misleading or incorrect)
You MUST NOT explain, justify, or output any extra characters, spaces, or words — only a single digit: either "0" or "1".
If the information cannot be verified, is ambiguous, or partially true/false, always classify it as class 0 (FALSE).
Return **ONLY** one character: "0" or "1".
<|end|>
<|user|>
{text}

# For any unclear or unverifiable samples, always classify them as class 0 (FALSE).
<|end|>
<|assistant|>"""





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

    print(f"Processing {len(texts)} samples...", flush=True)
    start = time.time()

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_prompts = [build_prompt(t) for t in batch_texts]

        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

        # Generate responses
        with torch.no_grad():
            batch_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True
            )

        # Parse model outputs
        for idx, output in enumerate(batch_outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            answer = decoded.split("<|assistant|>")[-1].strip()

            # Extract only the last number from the response
            numbers = re.findall(r'\d+', answer)
            if numbers:
                label = int(numbers[-1])
                # Nếu nhãn khác 0 và 1 thì gán là 0 (giả)
                if label not in [0, 1]:
                    label = 0
            else:
                label = 0  # fallback FAKE

            # Debug display only shows final output
            if DEBUG_MODE and len(outputs) < 10:
                print(f"\nSample {len(outputs)+1}: {label}")

            outputs.append(label)
            # Only keep the final label, no raw responses needed

        del inputs, batch_outputs
        torch.cuda.empty_cache()

    # Save results with only the labels
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
    print(f"✅ Done {len(outputs)} samples in {time.time()-start:.1f}s")
    print(f"📊 Label distribution (1=REAL, 0=FAKE):")
    print(f"   - REAL (1): {real_count} ({real_count/len(outputs)*100:.1f}%)")
    print(f"   - FAKE (0): {fake_count} ({fake_count/len(outputs)*100:.1f}%)")
    print(f"📂 Output saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
