import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 256

# ===============================
# LOAD DATA
# ===============================
df_train = pd.read_csv("label.csv")  # tập train có nhãn thật

df_val = pd.read_csv("val.csv")        # tập validation để tính ER

# ===============================
# DATASET
# ===============================
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

# ===============================
# LOAD MODELS (FROM COTRAIN)
# ===============================
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")

bert = AutoModelForSequenceClassification.from_pretrained("cotrain_bert").to(DEVICE)
roberta = AutoModelForSequenceClassification.from_pretrained("cotrain_roberta").to(DEVICE)

# ===============================
# DATALOADERS
# ===============================
train_bert = TextDataset(df_train["text"].tolist(), df_train["label"].tolist(), tokenizer_bert)
val_bert = TextDataset(df_val["text"].tolist(), df_val["label"].tolist(), tokenizer_bert)

train_roberta = TextDataset(df_train["text"].tolist(), df_train["label"].tolist(), tokenizer_roberta)
val_roberta = TextDataset(df_val["text"].tolist(), df_val["label"].tolist(), tokenizer_roberta)

train_loader_bert = DataLoader(train_bert, batch_size=BATCH_SIZE, shuffle=True)
train_loader_roberta = DataLoader(train_roberta, batch_size=BATCH_SIZE, shuffle=True)
val_loader_bert = DataLoader(val_bert, batch_size=BATCH_SIZE)
val_loader_roberta = DataLoader(val_roberta, batch_size=BATCH_SIZE)

opt_bert = AdamW(bert.parameters(), lr=LR)
opt_roberta = AdamW(roberta.parameters(), lr=LR)

# ===============================
# TRAIN & EVAL FUNCTIONS
# ===============================
def train_one(model, optimizer, loader, name):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Fine-tuning {name}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].unsqueeze(1).to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        probs = torch.sigmoid(outputs.logits)
        loss = F.binary_cross_entropy(probs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def evaluate(model, loader, name):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].unsqueeze(1).to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = total_correct / total
    er = 1 - acc
    print(f"[{name}] Accuracy: {acc*100:.2f}%, Error Rate: {er*100:.2f}%")
    return er


# ===============================
# TRAINING LOOP
# ===============================
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    loss_b = train_one(bert, opt_bert, train_loader_bert, "BERT")
    loss_r = train_one(roberta, opt_roberta, train_loader_roberta, "RoBERTa")

    print(f"\n→ Training Loss | BERT: {loss_b:.4f}, RoBERTa: {loss_r:.4f}")
    print("→ Evaluating on validation set...")
    er_b = evaluate(bert, val_loader_bert, "BERT")
    er_r = evaluate(roberta, val_loader_roberta, "RoBERTa")
    print(f"Epoch {epoch+1} ER: BERT={er_b*100:.2f}%, RoBERTa={er_r*100:.2f}%")

# ===============================
# SAVE MODELS
# ===============================
bert.save_pretrained("finetuned_bert")
roberta.save_pretrained("finetuned_roberta")
print("\n✅ Fine-tuning & validation completed.")
