import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
import numpy as np

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256

# ===============================
# LOAD DATA
# ===============================
df_data = pd.read_csv("weights_with_text.csv")  # chứa text, pseudo_label, lambda1, lambda2

texts = df_data["text"].tolist()
labels = df_data["pseudo_label"].tolist()
lambda1 = df_data["lambda1"].values
lambda2 = df_data["lambda2"].values

# ===============================
# DATASET
# ===============================
class TextDataset(Dataset):
    def __init__(self, texts, labels, lambda1, lambda2, tokenizer):
        self.texts = texts
        self.labels = labels
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item["text"] = self.texts[idx]  # Thêm text gốc để RoBERTa có thể tokenize lại
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        item["lambda1"] = torch.tensor(self.lambda1[idx], dtype=torch.float)
        item["lambda2"] = torch.tensor(self.lambda2[idx], dtype=torch.float)
        return item

# ===============================
# MODEL INIT
# ===============================
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")

bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(DEVICE)
roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1).to(DEVICE)

train_dataset = TextDataset(texts, labels, lambda1, lambda2, tokenizer_bert)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Optimizers
opt_bert = AdamW(bert.parameters(), lr=LR)
opt_roberta = AdamW(roberta.parameters(), lr=LR)

# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(EPOCHS):
    bert.train()
    roberta.train()
    total_loss_bert, total_loss_roberta = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].unsqueeze(1).to(DEVICE)
        l1 = batch["lambda1"].unsqueeze(1).to(DEVICE)
        l2 = batch["lambda2"].unsqueeze(1).to(DEVICE)

        # --- BERT update (weighted by λ2) ---
        out_bert = bert(input_ids=input_ids, attention_mask=attn_mask)
        prob_bert = torch.sigmoid(out_bert.logits)
        loss_bert = torch.mean(l2 * F.binary_cross_entropy(prob_bert, labels, reduction='none'))
        opt_bert.zero_grad()
        loss_bert.backward()
        opt_bert.step()

        # --- RoBERTa update (weighted by λ1) ---
        input_roberta = tokenizer_roberta(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        
        out_roberta = roberta(**input_roberta)
        prob_roberta = torch.sigmoid(out_roberta.logits)
        loss_roberta = torch.mean(l1 * F.binary_cross_entropy(prob_roberta, labels, reduction='none'))
        opt_roberta.zero_grad()
        loss_roberta.backward()
        opt_roberta.step()

        # Update metrics
        total_loss_bert += loss_bert.item()
        total_loss_roberta += loss_roberta.item()

        # Update progress bar
        loop.set_postfix({
            'bert_loss': loss_bert.item(),
            'roberta_loss': loss_roberta.item()
        })

    # Print epoch results
    avg_loss_bert = total_loss_bert / len(train_loader)
    avg_loss_roberta = total_loss_roberta / len(train_loader)
    print(f"\nEpoch {epoch+1}:")
    print(f"BERT avg loss: {avg_loss_bert:.4f}")
    print(f"RoBERTa avg loss: {avg_loss_roberta:.4f}")

# ===============================
# SAVE MODELS
# ===============================
print("\nSaving models...")
bert.save_pretrained("cotrain_bert")
roberta.save_pretrained("cotrain_roberta")
print("Co-training completed!")