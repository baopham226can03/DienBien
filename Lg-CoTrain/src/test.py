import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Đường dẫn =====
BERT_DIR = "./finetuned_bert"
ROBERTA_DIR = "./finetuned_roberta"
TEST_FILE = "./test.csv"

# ===== Mô hình gốc =====
BERT_BASE = "bert-base-uncased"
ROBERTA_BASE = "roberta-base"

# ===== Load test =====
df = pd.read_csv(TEST_FILE)
texts = df["text"].tolist()
labels = df["label"].tolist()

def evaluate_model(model_dir, base_model, name):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(DEVICE)
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            preds.append(pred)

    acc = accuracy_score(labels, preds)
    er = 1 - acc
    print(f"{name} - Accuracy: {acc*100:.2f}% | Error Rate: {er*100:.2f}%")
    return er

if __name__ == "__main__":
    print("Evaluating fine-tuned models...\n")
    er_bert = evaluate_model(BERT_DIR, BERT_BASE, "BERT")
    er_roberta = evaluate_model(ROBERTA_DIR, ROBERTA_BASE, "RoBERTa")

    print("\n===============================")
    print(f"📊 Final Results:")
    print(f"BERT Error Rate: {er_bert*100:.2f}%")
    print(f"RoBERTa Error Rate: {er_roberta*100:.2f}%")
    print("===============================")
