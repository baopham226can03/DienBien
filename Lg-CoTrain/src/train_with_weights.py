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
BATCH_SIZE = 32  # As specified in the paper
MAX_LEN = 256
LR = 2e-5       # As specified in the paper

# Training parameters as per paper
COTRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5

# ===============================
# DATASET CLASSES
# ===============================
class WeightedDataset(Dataset):
    def __init__(self, texts, labels, lambda1, lambda2, tokenizer):
        self.texts = texts
        self.labels = labels
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tokenizer = tokenizer

    def update_weights(self, new_lambda1, new_lambda2):
        """Update weight values for all samples"""
        self.lambda1 = new_lambda1
        self.lambda2 = new_lambda2

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
        item["text"] = self.texts[idx]  # For RoBERTa retokenization
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        item["lambda1"] = torch.tensor(self.lambda1[idx], dtype=torch.float)
        item["lambda2"] = torch.tensor(self.lambda2[idx], dtype=torch.float)
        return item

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
# TRAINING FUNCTIONS
# ===============================
def train_cotrain_epoch(bert, roberta, opt_bert, opt_roberta, loader, tokenizer_roberta):
    bert.train()
    roberta.train()
    total_loss_bert, total_loss_roberta = 0, 0

    loop = tqdm(loader, desc="Co-training")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].unsqueeze(1).to(DEVICE)
        l1 = batch["lambda1"].unsqueeze(1).to(DEVICE)
        l2 = batch["lambda2"].unsqueeze(1).to(DEVICE)

        # BERT update (weighted by λ2)
        with torch.cuda.amp.autocast():
            out_bert = bert(input_ids=input_ids, attention_mask=attn_mask)
            loss_bert = torch.mean(l2 * F.binary_cross_entropy_with_logits(out_bert.logits, labels, reduction='none'))
        opt_bert.zero_grad()
        scaler.scale(loss_bert).backward()
        scaler.step(opt_bert)
        scaler.update()

        # RoBERTa update (weighted by λ1)
        input_roberta = tokenizer_roberta(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.cuda.amp.autocast():
            out_roberta = roberta(**input_roberta)
            loss_roberta = torch.mean(l1 * F.binary_cross_entropy_with_logits(out_roberta.logits, labels, reduction='none'))
        opt_roberta.zero_grad()
        scaler.scale(loss_roberta).backward()
        scaler.step(opt_roberta)
        scaler.update()

        total_loss_bert += loss_bert.item()
        total_loss_roberta += loss_roberta.item()
        loop.set_postfix({
            'bert_loss': loss_bert.item(),
            'roberta_loss': loss_roberta.item()
        })

    return total_loss_bert / len(loader), total_loss_roberta / len(loader)

def train_finetune_epoch(model, optimizer, loader, name):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Fine-tuning {name}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].unsqueeze(1).to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            loss = F.binary_cross_entropy_with_logits(outputs.logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                probs = torch.sigmoid(outputs.logits.float())
                preds = (probs > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = total_correct / total
    er = 1 - acc
    print(f"[{name}] Accuracy: {acc*100:.2f}%, Error Rate: {er*100:.2f}%")
    return er

# ===============================
# DATA PREPARATION
# ===============================
def prepare_weighted_data():
    print("Preparing weighted training data...")
    # Load weights and pseudo-labeled data
    df_weights = pd.read_csv("weights.csv")
    df_pseudo = pd.read_csv("pseudo.csv")
    
    # Ensure the lengths match
    assert len(df_weights) == len(df_pseudo), "Weights and pseudo labels must have the same length"
    
    # Combine the data
    df_combined = pd.DataFrame({
        'text': df_pseudo['text'],
        'pseudo_label': df_pseudo['pseudo_label'],
        'lambda1': df_weights['lambda1'],
        'lambda2': df_weights['lambda2']
    })
    
    # Save combined data
    df_combined.to_csv("weights_with_text.csv", index=False)
    print("✓ Created weights_with_text.csv with combined data")
    return df_combined

# ===============================
# MAIN TRAINING PROCESS
# ===============================
def main():
    print("Initializing models and tokenizers...")
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")

    # Set tokenizer parallelism explicitly
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize models
    bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(DEVICE)
    roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1).to(DEVICE)

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Prepare and load weighted data for co-training
    print("\nPreparing and loading weighted data for co-training...")
    df_weighted = prepare_weighted_data()
    train_dataset = WeightedDataset(
        df_weighted["text"].tolist(),
        df_weighted["pseudo_label"].tolist(),
        df_weighted["lambda1"].values,
        df_weighted["lambda2"].values,
        tokenizer_bert
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Disable multiprocessing for stability

    # Optimizers
    opt_bert = AdamW(bert.parameters(), lr=LR)
    opt_roberta = AdamW(roberta.parameters(), lr=LR)

    # ======== CO-TRAINING PHASE ========
    print("\n=== Starting Co-training Phase ===")
    best_loss_bert = float('inf')
    best_loss_roberta = float('inf')
    patience_counter = 0
    best_bert_state = None
    best_roberta_state = None

    # Get initial probabilities (from initial models)
    bert.eval()
    roberta.eval()
    initial_probs_bert = []
    initial_probs_roberta = []
    
    print("Getting initial probabilities...")
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Initial probs"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            
            # Get BERT probs
            outputs_bert = bert(input_ids=input_ids, attention_mask=attn_mask)
            probs_bert = torch.sigmoid(outputs_bert.logits).cpu().numpy()
            initial_probs_bert.extend(probs_bert.squeeze())
            
            # Get RoBERTa probs (need to retokenize)
            inputs_roberta = tokenizer_roberta(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)
            outputs_roberta = roberta(**inputs_roberta)
            probs_roberta = torch.sigmoid(outputs_roberta.logits).cpu().numpy()
            initial_probs_roberta.extend(probs_roberta.squeeze())
    
    # Convert to numpy arrays
    initial_probs_bert = np.array(initial_probs_bert)
    initial_probs_roberta = np.array(initial_probs_roberta)
    
    # Initialize probability history with initial probs
    all_probs_bert = [initial_probs_bert]
    all_probs_roberta = [initial_probs_roberta]
    
    # Current lambda values
    current_lambda1 = df_weighted["lambda1"].values
    current_lambda2 = df_weighted["lambda2"].values

    for epoch in range(COTRAIN_EPOCHS):
        print(f"\nCo-training Epoch {epoch+1}/{COTRAIN_EPOCHS}")
        
        # Update DataLoader with current lambdas
        train_dataset.update_weights(current_lambda1, current_lambda2)
        
        # Train one epoch
        bert.train()
        roberta.train()
        total_loss_bert = 0
        total_loss_roberta = 0
        epoch_probs_bert = []
        epoch_probs_roberta = []
        
        for batch in tqdm(train_loader, desc="Co-training"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].unsqueeze(1).to(DEVICE)
            l1 = batch["lambda1"].unsqueeze(1).to(DEVICE)
            l2 = batch["lambda2"].unsqueeze(1).to(DEVICE)

            # BERT update (weighted by λ2)
            with torch.cuda.amp.autocast():
                outputs_bert = bert(input_ids=input_ids, attention_mask=attn_mask)
                loss_bert = torch.mean(l2 * F.binary_cross_entropy_with_logits(outputs_bert.logits, labels, reduction='none'))
            
            opt_bert.zero_grad()
            scaler.scale(loss_bert).backward()
            scaler.step(opt_bert)
            scaler.update()
            
            # Store BERT probs
            with torch.no_grad():
                probs_bert = torch.sigmoid(outputs_bert.logits.float())
                epoch_probs_bert.extend(probs_bert.detach().cpu().numpy().squeeze())

            # RoBERTa update (weighted by λ1)
            inputs_roberta = tokenizer_roberta(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.cuda.amp.autocast():
                outputs_roberta = roberta(**inputs_roberta)
                loss_roberta = torch.mean(l1 * F.binary_cross_entropy_with_logits(outputs_roberta.logits, labels, reduction='none'))
            
            opt_roberta.zero_grad()
            scaler.scale(loss_roberta).backward()
            scaler.step(opt_roberta)
            scaler.update()
            
            # Store RoBERTa probs
            with torch.no_grad():
                probs_roberta = torch.sigmoid(outputs_roberta.logits.float())
                epoch_probs_roberta.extend(probs_roberta.detach().cpu().numpy().squeeze())
            
            total_loss_bert += loss_bert.item()
            total_loss_roberta += loss_roberta.item()

        # Calculate average losses
        avg_loss_bert = total_loss_bert / len(train_loader)
        avg_loss_roberta = total_loss_roberta / len(train_loader)
        print(f"BERT loss: {avg_loss_bert:.4f}, RoBERTa loss: {avg_loss_roberta:.4f}")

        # Add new probabilities to history
        all_probs_bert.append(np.array(epoch_probs_bert))
        all_probs_roberta.append(np.array(epoch_probs_roberta))
        
        # Update confidence and variability metrics
        def compute_metrics(probs_history):
            probs_array = np.array(probs_history)
            confidence = np.mean(probs_array, axis=0)  # Eq. (1)
            variability = np.std(probs_array, axis=0)  # Eq. (2)
            return confidence, variability
        
        # Calculate new metrics including initial probs
        c1, v1 = compute_metrics(all_probs_bert)
        c2, v2 = compute_metrics(all_probs_roberta)
        
        # Update lambda values for next epoch
        current_lambda1 = c1 + v1  # Eq. (3)
        current_lambda2 = c2 - v2  # Eq. (4)
        
        # Early stopping check
        current_loss = avg_loss_bert + avg_loss_roberta
        if current_loss < (best_loss_bert + best_loss_roberta):
            best_loss_bert = avg_loss_bert
            best_loss_roberta = avg_loss_roberta
            patience_counter = 0
            best_bert_state = bert.state_dict().copy()
            best_roberta_state = roberta.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # Load best states from co-training
    if best_bert_state is not None:
        bert.load_state_dict(best_bert_state)
    if best_roberta_state is not None:
        roberta.load_state_dict(best_roberta_state)

    # Save co-trained models
    print("\nSaving co-trained models...")
    bert.save_pretrained("cotrain_bert")
    roberta.save_pretrained("cotrain_roberta")

    # ======== FINE-TUNING PHASE ========
    print("\n=== Starting Fine-tuning Phase ===")
    
    # Load labeled data for fine-tuning
    print("Loading labeled data...")
    df_train = pd.read_csv("label.csv")
    df_val = pd.read_csv("val.csv")

    # Create datasets and dataloaders for fine-tuning
    train_bert = TextDataset(df_train["text"].tolist(), df_train["label"].tolist(), tokenizer_bert)
    val_bert = TextDataset(df_val["text"].tolist(), df_val["label"].tolist(), tokenizer_bert)
    train_roberta = TextDataset(df_train["text"].tolist(), df_train["label"].tolist(), tokenizer_roberta)
    val_roberta = TextDataset(df_val["text"].tolist(), df_val["label"].tolist(), tokenizer_roberta)

    train_loader_bert = DataLoader(train_bert, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    train_loader_roberta = DataLoader(train_roberta, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_bert = DataLoader(val_bert, batch_size=BATCH_SIZE, num_workers=0)
    val_loader_roberta = DataLoader(val_roberta, batch_size=BATCH_SIZE, num_workers=0)

    # Reset optimizers for fine-tuning
    opt_bert = AdamW(bert.parameters(), lr=LR)
    opt_roberta = AdamW(roberta.parameters(), lr=LR)

    # Fine-tuning with early stopping
    best_er_bert = float('inf')
    best_er_roberta = float('inf')
    patience_counter_bert = 0
    patience_counter_roberta = 0
    best_bert_state = None
    best_roberta_state = None

    for epoch in range(FINETUNE_EPOCHS):
        print(f"\n=== Fine-tuning Epoch {epoch+1}/{FINETUNE_EPOCHS} ===")
        
        # Train
        loss_bert = train_finetune_epoch(bert, opt_bert, train_loader_bert, "BERT")
        loss_roberta = train_finetune_epoch(roberta, opt_roberta, train_loader_roberta, "RoBERTa")
        print(f"Training Loss | BERT: {loss_bert:.4f}, RoBERTa: {loss_roberta:.4f}")

        # Evaluate
        print("Evaluating on validation set...")
        er_bert = evaluate(bert, val_loader_bert, "BERT")
        er_roberta = evaluate(roberta, val_loader_roberta, "RoBERTa")

        # Early stopping for BERT
        if er_bert < best_er_bert:
            best_er_bert = er_bert
            patience_counter_bert = 0
            best_bert_state = bert.state_dict().copy()
        else:
            patience_counter_bert += 1

        # Early stopping for RoBERTa
        if er_roberta < best_er_roberta:
            best_er_roberta = er_roberta
            patience_counter_roberta = 0
            best_roberta_state = roberta.state_dict().copy()
        else:
            patience_counter_roberta += 1

        # Check if both models should stop
        if (patience_counter_bert >= EARLY_STOPPING_PATIENCE and 
            patience_counter_roberta >= EARLY_STOPPING_PATIENCE):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Load best states from fine-tuning
    if best_bert_state is not None:
        bert.load_state_dict(best_bert_state)
    if best_roberta_state is not None:
        roberta.load_state_dict(best_roberta_state)

    # Save final fine-tuned models
    print("\nSaving fine-tuned models...")
    bert.save_pretrained("finetuned_bert")
    roberta.save_pretrained("finetuned_roberta")
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()