import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm

# ======== CONFIG ========
BERT_MODEL = "bert-base-uncased"
ROBERTA_MODEL = "roberta-base"
NUM_EPOCHS_LABELED = 100    # số epoch finetune trên tập labeled
NUM_EPOCHS_PSEUDO = 10      # số epoch train pseudo để sinh weight
EARLY_STOPPING_PATIENCE = 5  # số epoch chờ trước khi early stopping
BATCH_SIZE = 32
MAX_LEN = 256
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======== LOAD DATA ========
df_labeled = pd.read_csv("labeled.csv")         # cột: text, label
df_pseudo = pd.read_csv("pseudo.csv")  # cột: text, pseudo_label

# Split labeled data into two equal parts maintaining class distribution
def split_stratified(df):
    # Get indices for each class
    class_0_idx = df[df['label'] == 0].index
    class_1_idx = df[df['label'] == 1].index
    
    # Randomly split each class into two equal parts
    class_0_split1 = np.random.choice(class_0_idx, size=len(class_0_idx)//2, replace=False)
    class_1_split1 = np.random.choice(class_1_idx, size=len(class_1_idx)//2, replace=False)
    
    # Create first subset
    split1_idx = np.concatenate([class_0_split1, class_1_split1])
    df_split1 = df.loc[split1_idx].reset_index(drop=True)
    
    # Create second subset
    split2_idx = df.index.difference(split1_idx)
    df_split2 = df.loc[split2_idx].reset_index(drop=True)
    
    return df_split1, df_split2

# Split labeled data for each model
df_labeled_bert, df_labeled_roberta = split_stratified(df_labeled)
print(f"Split sizes - BERT: {len(df_labeled_bert)}, RoBERTa: {len(df_labeled_roberta)}")

tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL)
tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_MODEL)

# ======== DATASET ========
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, label_col):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
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
            "labels": torch.tensor(self.labels[i], dtype=torch.float)
        }

# ======== DATA LOADERS ========
ds_labeled_bert = TextDataset(df_labeled_bert, tokenizer_bert, "text", "label")
ds_labeled_roberta = TextDataset(df_labeled_roberta, tokenizer_roberta, "text", "label")
loader_labeled_bert = DataLoader(ds_labeled_bert, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_labeled_roberta = DataLoader(ds_labeled_roberta, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

ds_pseudo_bert = TextDataset(df_pseudo, tokenizer_bert, "text", "pseudo_label")
ds_pseudo_roberta = TextDataset(df_pseudo, tokenizer_roberta, "text", "pseudo_label")
loader_pseudo_bert = DataLoader(ds_pseudo_bert, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)  # No shuffle for consistent probability matrix
loader_pseudo_roberta = DataLoader(ds_pseudo_roberta, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ======== MODELS ========
model_bert = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=1).to(DEVICE)
model_roberta = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL, num_labels=1).to(DEVICE)

optimizer_bert = AdamW(model_bert.parameters(), lr=LR)
optimizer_roberta = AdamW(model_roberta.parameters(), lr=LR)

scaler = torch.cuda.amp.GradScaler()

# ======== TRAIN FUNCTION ========
def train_model(model, optimizer, loader, name, num_epochs):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"{name} Epoch {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].unsqueeze(1).float().to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attn)
                logits = outputs.logits
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(loader)
        print(f"✅ {name} Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"{name}_finetuned.pt")
    return model

# ======== STEP 1: FINETUNE ON LABELED DATA ========
print("=== Step 1: Finetuning on labeled data ===")
model_bert = train_model(model_bert, optimizer_bert, loader_labeled_bert, "bert", NUM_EPOCHS_LABELED)
model_roberta = train_model(model_roberta, optimizer_roberta, loader_labeled_roberta, "roberta", NUM_EPOCHS_LABELED)

# Reset optimizer for pseudo training
optimizer_bert = AdamW(model_bert.parameters(), lr=LR)
optimizer_roberta = AdamW(model_roberta.parameters(), lr=LR)

# ======== STEP 2: TRAIN ON PSEUDO + COLLECT PROBABILITIES ========
def collect_prob_matrix(model, loader, num_samples):
    """Collect probability matrix P for all samples"""
    model.eval()
    prob_matrix = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting probabilities"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attn)
            logits = outputs.logits.float()  # Convert to float32 for stability
            probs = torch.sigmoid(logits)
            prob_matrix.extend(probs.cpu().numpy().squeeze())
    
    return np.array(prob_matrix)

def train_and_collect(model, optimizer, train_loader, eval_loader, name, num_epochs):
    """Train model and collect probability matrix P across epochs"""
    model.train()
    all_epoch_probs = []  # Will store P matrix (epochs × samples)
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Get initial probabilities before training
    print(f"\nCollecting initial probabilities for {name}...")
    initial_probs = collect_prob_matrix(model, eval_loader, len(df_pseudo))
    all_epoch_probs.append(initial_probs)
    
    for epoch in range(num_epochs):
        # Training
        total_loss = 0
        model.train()
        for batch in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].unsqueeze(1).float().to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attn)
                logits = outputs.logits
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        # Collect probabilities for all pseudo-labeled samples
        model.eval()
        epoch_probs = collect_prob_matrix(model, eval_loader, len(df_pseudo))
        all_epoch_probs.append(epoch_probs)
                
        avg_loss = total_loss/len(train_loader)
        print(f"✅ {name} Pseudo Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), f"{name}_pseudo_ep{epoch+1}.pt")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch_probs = epoch_probs
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return np.array(all_epoch_probs)

print("\n=== Step 2: Training on pseudo-labeled data and collecting probs ===")
prob_bert = train_and_collect(model_bert, optimizer_bert, loader_pseudo_bert, loader_pseudo_bert, "bert", NUM_EPOCHS_PSEUDO)
prob_roberta = train_and_collect(model_roberta, optimizer_roberta, loader_pseudo_roberta, loader_pseudo_roberta, "roberta", NUM_EPOCHS_PSEUDO)

# ======== STEP 3: COMPUTE λ1, λ2 ========
def compute_confidence_variability(prob_matrix):
    """
    Compute confidence and variability according to paper equations (1) and (2)
    prob_matrix: shape (num_epochs+1, num_samples) including initial probabilities
    """
    # Confidence: mean of predicted probabilities across all epochs (Eq. 1)
    confidence = np.mean(prob_matrix, axis=0)
    
    # Variability: std of predicted probabilities across all epochs (Eq. 2)
    # Considering initial probabilities as part of the calculation
    squared_diff = np.square(prob_matrix - confidence.reshape(1, -1))
    variability = np.sqrt(np.mean(squared_diff, axis=0))
    
    return confidence, variability

# Compute confidence and variability for each model
c1, v1 = compute_confidence_variability(prob_bert)  # For θ1 (BERT)
c2, v2 = compute_confidence_variability(prob_roberta)  # For θ2 (RoBERTa)

# Compute importance weights (Eq. 3 and 4)
lambda1 = c1 + v1  # Weight for θ2 based on θ1's dynamics
lambda2 = c2 - v2  # Weight for θ1 based on θ2's dynamics

# Normalize weights to [0,1] range to prevent extreme values
lambda1 = (lambda1 - lambda1.min()) / (lambda1.max() - lambda1.min() + 1e-8)
lambda2 = (lambda2 - lambda2.min()) / (lambda2.max() - lambda2.min() + 1e-8)

# Create output DataFrame
out = pd.DataFrame({
    'text': df_pseudo['text'],
    'pseudo_label': df_pseudo['pseudo_label'],
    'c_bert': c1, 
    'v_bert': v1,
    'c_roberta': c2, 
    'v_roberta': v2,
    'lambda1': lambda1,  # Used to weight samples for RoBERTa
    'lambda2': lambda2   # Used to weight samples for BERT
})

# Save results
out.to_csv("weights.csv", index=False)

print("\n✅ Done. weights.csv saved with λ₁, λ₂.")
print("✅ Finetuned models saved as bert_finetuned.pt and roberta_finetuned.pt")
