import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LEN = 256

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def get_predictions(model, loader, tokenizer=None, is_roberta=False):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if is_roberta and tokenizer:
                # Use BERT tokenization but convert to RoBERTa format
                input_ids = batch["input_ids"]
                attn_mask = batch["attention_mask"]
                labels = batch["label"].to(DEVICE)
                
                # Convert BERT tokens to text
                decoded_texts = tokenizer_bert.batch_decode(input_ids, skip_special_tokens=True)
                
                # Re-encode with RoBERTa
                roberta_inputs = tokenizer(
                    decoded_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt"
                ).to(DEVICE)
                
                outputs = model(**roberta_inputs)
            else:
                # Use BERT tokenization
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)

            with torch.amp.autocast('cuda'):
                logits = outputs.logits
                probs = torch.sigmoid(logits.float())

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)

def print_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"\n=== {model_name} Metrics ===")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-score: {f1*100:.2f}%")
    print(f"Error Rate: {(1-accuracy)*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'error_rate': 1-accuracy
    }

def main():
    print("Loading models and tokenizers...")
    # Load tokenizers
    global tokenizer_bert  # Make it accessible in get_predictions
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")

    # Load fine-tuned models from absolute paths
    bert = AutoModelForSequenceClassification.from_pretrained("/root/DienBien/Lg-CoTrain/src/finetuned_bert").to(DEVICE)
    roberta = AutoModelForSequenceClassification.from_pretrained("/root/DienBien/Lg-CoTrain/src/finetuned_roberta").to(DEVICE)

    print("\nLoading test data...")
    df_test = pd.read_csv("test.csv")
    
    # Create test datasets
    test_dataset = TextDataset(
        df_test["text"].tolist(),
        df_test["label"].tolist(),
        tokenizer_bert
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nGetting predictions from both models...")
    # Get predictions from both models
    bert_probs, true_labels = get_predictions(bert, test_loader)
    roberta_probs, _ = get_predictions(
        roberta, test_loader, 
        tokenizer=tokenizer_roberta, 
        is_roberta=True
    )

    # Individual model predictions
    bert_preds = (bert_probs.squeeze() > 0.5).astype(int)
    roberta_preds = (roberta_probs.squeeze() > 0.5).astype(int)

    # Ensemble predictions (average probabilities)
    ensemble_probs = (bert_probs + roberta_probs) / 2
    ensemble_preds = (ensemble_probs.squeeze() > 0.5).astype(int)

    # Calculate and print metrics for all models
    bert_metrics = print_metrics(true_labels, bert_preds, "BERT")
    roberta_metrics = print_metrics(true_labels, roberta_preds, "RoBERTa")
    ensemble_metrics = print_metrics(true_labels, ensemble_preds, "Ensemble (BERT + RoBERTa)")

    # Save predictions and metrics
    results_df = pd.DataFrame({
        'text': df_test["text"],
        'true_label': true_labels,
        'bert_pred': bert_preds,
        'roberta_pred': roberta_preds,
        'ensemble_pred': ensemble_preds,
        'bert_prob': bert_probs.squeeze(),
        'roberta_prob': roberta_probs.squeeze(),
        'ensemble_prob': ensemble_probs.squeeze()
    })
    results_df.to_csv("test_predictions.csv", index=False)

    # Save metrics
    metrics = {
        'bert': bert_metrics,
        'roberta': roberta_metrics,
        'ensemble': ensemble_metrics
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("test_metrics.csv")

    print("\nResults saved to:")
    print("- test_predictions.csv (all predictions and probabilities)")
    print("- test_metrics.csv (accuracy, precision, recall, F1 scores)")

if __name__ == "__main__":
    main()