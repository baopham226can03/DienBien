import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm

# ======== CONFIG ========
BERT_MODEL = "bert-base-uncased"
ROBERTA_MODEL = "roberta-base"
NUM_EPOCHS = 5
BATCH_SIZE = 16
MAX_LEN = 256
LR = 2e-5
DEVICE = "cuda"

# ======== DATA ========
df_pseudo = pd.read_csv("unlabel.csv")  # cột: text, pseudo_label

tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL)
tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_MODEL)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = df["pseudo_label"].tolist()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[i], dtype=torch.long)
        }

ds_bert = TextDataset(df_pseudo, tokenizer_bert)
ds_roberta = TextDataset(df_pseudo, tokenizer_roberta)
loader_bert = DataLoader(ds_bert, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_roberta = DataLoader(ds_roberta, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ======== MODELS ========
model_bert = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2).to(DEVICE)
model_roberta = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL, num_labels=2).to(DEVICE)

optimizer_bert = AdamW(model_bert.parameters(), lr=LR)
optimizer_roberta = AdamW(model_roberta.parameters(), lr=LR)

# ======== TRAIN + COLLECT PROBS ========
scaler = torch.cuda.amp.GradScaler()
prob_bert, prob_roberta = [], []

def train_and_collect(model, optimizer, loader, name):
    model.train()
    epoch_probs = []
    for batch in tqdm(loader, desc=f"{name} training", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attn, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().numpy()
            epoch_probs.extend(probs)
    torch.save(model.state_dict(), f"{name}_epoch.pt")
    return np.array(epoch_probs)

for epoch in range(NUM_EPOCHS):
    p1 = train_and_collect(model_bert, optimizer_bert, loader_bert, f"bert_ep{epoch+1}")
    p2 = train_and_collect(model_roberta, optimizer_roberta, loader_roberta, f"roberta_ep{epoch+1}")
    prob_bert.append(p1)
    prob_roberta.append(p2)

# ======== TÍNH λ1, λ2 ========
prob_bert = np.stack(prob_bert, axis=0)
prob_roberta = np.stack(prob_roberta, axis=0)

def compute_conf_var(p):
    c = np.mean(p, axis=0)
    v = np.var(p, axis=0)
    return c, v

c1, v1 = compute_conf_var(prob_bert)
c2, v2 = compute_conf_var(prob_roberta)

lambda1 = c1 + v1
lambda2 = c2 - v2

out = pd.DataFrame({
    "c_bert": c1, "v_bert": v1,
    "c_roberta": c2, "v_roberta": v2,
    "lambda1": lambda1, "lambda2": lambda2
})
out.to_csv("weights.csv", index=False)
print("✅ Done. weights.csv saved with λ1, λ2.")
