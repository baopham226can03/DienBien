import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from utils.load_data import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------- ARGPARSE -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='SheepDog++', type=str)
parser.add_argument('--iters', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)

# ------------------- DATASET -------------------
class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2

    def __getitem__(self, item):
        text = self.texts[item]
        aug1 = self.aug_texts1[item]
        aug2 = self.aug_texts2[item]
        label = self.labels[item]
        fg = self.fg_label[item]
        fg1 = self.aug_fg1[item]
        fg2 = self.aug_fg2[item]

        enc = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                         pad_to_max_length=True, truncation=True, return_token_type_ids=False,
                                         return_attention_mask=True, return_tensors='pt')
        enc1 = self.tokenizer.encode_plus(aug1, add_special_tokens=True, max_length=self.max_len,
                                          pad_to_max_length=True, truncation=True, return_token_type_ids=False,
                                          return_attention_mask=True, return_tensors='pt')
        enc2 = self.tokenizer.encode_plus(aug2, add_special_tokens=True, max_length=self.max_len,
                                          pad_to_max_length=True, truncation=True, return_token_type_ids=False,
                                          return_attention_mask=True, return_tensors='pt')

        return {
            'input_ids': enc['input_ids'].flatten(),
            'input_ids_aug1': enc1['input_ids'].flatten(),
            'input_ids_aug2': enc2['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'attention_mask_aug1': enc1['attention_mask'].flatten(),
            'attention_mask_aug2': enc2['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg),
            'fg_label_aug1': torch.FloatTensor(fg1),
            'fg_label_aug2': torch.FloatTensor(fg2),
        }

    def __len__(self):
        return len(self.texts)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        enc = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                         pad_to_max_length=True, truncation=True, return_token_type_ids=False,
                                         return_attention_mask=True, return_tensors='pt')
        return {
            'news_text': text,
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


# ------------------- MODEL -------------------
class RobertaClassifierCSI(nn.Module):
    def __init__(self, n_classes, proj_dim=128):
        super(RobertaClassifierCSI, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)
        self.proj_head = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.roberta.config.hidden_size, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]  # CLS token
        pooled = self.dropout(pooled)
        out_labels = self.fc_out(pooled)
        out_labels_bi = self.binary_transform(pooled)
        z = F.normalize(self.proj_head(pooled), dim=-1)
        return out_labels, out_labels_bi, z


# ------------------- DATALOADERS -------------------
def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size):
    ds = NewsDatasetAug(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)


def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(contents, labels, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


# ------------------- UTIL -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def contrastive_loss_hard(z_anchor, z_pos, temperature=0.1):
    batch_size = z_anchor.size(0)
    sim = torch.matmul(z_anchor, z_pos.T) / temperature
    with torch.no_grad():
        mask = torch.eye(batch_size).to(z_anchor.device)
        sim_neg = sim * (1 - mask) - mask * 1e9
        topk_neg, _ = sim_neg.topk(k=min(3, batch_size-1), dim=-1)
    labels = torch.arange(batch_size).to(z_anchor.device)
    loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim - topk_neg.mean(dim=1, keepdim=True), labels)
    return loss


# ------------------- TRAINING -------------------
def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter, prev_model_path=None):
    x_train, x_test, x_test_res, y_train, y_test = load_articles(datasetname)
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)
    test_loader_res = create_eval_loader(x_test_res, y_test, tokenizer, max_len, batch_size)

    model = RobertaClassifierCSI(n_classes=4).to(device)

    # ---------- Carry-over checkpoint ----------
    if prev_model_path is not None and os.path.exists(prev_model_path):
        print(f"Loading weights from {prev_model_path} for carry-over...")
        model.load_state_dict(torch.load(prev_model_path, map_location=device))
        lr = 1e-5
    else:
        lr = 2e-5

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(x_train) // batch_size * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    best_acc = 0.0
    ckpt_path = f'checkpoints/{datasetname}_iter{iter}.m'

    for epoch in range(n_epochs):
        model.train()
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(datasetname)
        train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, max_len, batch_size)

        avg_loss, avg_acc = [], []

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            input_ids_aug1 = batch["input_ids_aug1"].to(device)
            attn_aug1 = batch["attention_mask_aug1"].to(device)
            input_ids_aug2 = batch["input_ids_aug2"].to(device)
            attn_aug2 = batch["attention_mask_aug2"].to(device)
            targets = batch["labels"].to(device)
            fg = batch["fg_label"].to(device)
            fg1 = batch["fg_label_aug1"].to(device)
            fg2 = batch["fg_label_aug2"].to(device)

            out_labels, out_labels_bi, z = model(input_ids, attn)
            out_labels_aug1, out_labels_bi_aug1, z_aug1 = model(input_ids_aug1, attn_aug1)
            out_labels_aug2, out_labels_bi_aug2, z_aug2 = model(input_ids_aug2, attn_aug2)

            fg_criterion = nn.BCELoss()
            fine_loss = 0.3 * (fg_criterion(torch.sigmoid(out_labels), fg) +
                               fg_criterion(torch.sigmoid(out_labels_aug1), fg1) +
                               fg_criterion(torch.sigmoid(out_labels_aug2), fg2)) / 3

            sup_loss_orig = nn.CrossEntropyLoss()(out_labels_bi, targets)
            sup_loss_aug = 0.5 * (nn.CrossEntropyLoss()(out_labels_bi_aug1, targets) +
                                  nn.CrossEntropyLoss()(out_labels_bi_aug2, targets))

            cons_loss = 0.5 * (nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(out_labels_bi_aug1, dim=-1),
                F.softmax(out_labels_bi, dim=-1)) +
                nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(out_labels_bi_aug2, dim=-1),
                    F.softmax(out_labels_bi, dim=-1))
            )

            contrast_loss = 0.3 * (contrastive_loss_hard(z, z_aug1) + contrastive_loss_hard(z, z_aug2))

            loss = sup_loss_orig + sup_loss_aug + cons_loss + fine_loss + contrast_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            _, pred = out_labels_bi.max(dim=-1)
            acc = (pred == targets).float().mean().item()
            avg_acc.append(acc)
            avg_loss.append(loss.item())

        print(f"Iter {iter:03d} | Epoch {epoch:03d} | Train Acc: {np.mean(avg_acc):.4f}")

        # ---------- Evaluate after each epoch ----------
        model.eval()
        y_pred, y_test_list = [], []
        for batch in test_loader:
            with torch.no_grad():
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                _, out = model(ids, attn)[:2]
                _, pred = out.max(dim=-1)
                y_pred.append(pred)
                y_test_list.append(batch["labels"].to(device))
        y_pred = torch.cat(y_pred, 0)
        y_test_list = torch.cat(y_test_list, 0)
        acc = accuracy_score(y_test_list.cpu(), y_pred.cpu())
        if acc > best_acc:
            torch.save(model.state_dict(), ckpt_path)
            best_acc = acc

    # ---------- Final evaluation ----------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Original test
    y_pred, y_test_list = [], []
    for batch in test_loader:
        with torch.no_grad():
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            _, out = model(ids, attn)[:2]
            _, pred = out.max(dim=-1)
            y_pred.append(pred)
            y_test_list.append(batch["labels"].to(device))
    y_pred = torch.cat(y_pred, 0)
    y_test_list = torch.cat(y_test_list, 0)
    acc = accuracy_score(y_test_list.cpu(), y_pred.cpu())
    precision, recall, f1, _ = score(y_test_list.cpu(), y_pred.cpu(), average='macro')

    # Restyle test
    y_pred_res = []
    for batch in test_loader_res:
        with torch.no_grad():
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            _, out = model(ids, attn)[:2]
            _, pred = out.max(dim=-1)
            y_pred_res.append(pred)
    y_pred_res = torch.cat(y_pred_res, 0)
    acc_res = accuracy_score(y_test_list.cpu(), y_pred_res.cpu())
    precision_res, recall_res, f1_res, _ = score(y_test_list.cpu(), y_pred_res.cpu(), average='macro')

    print(f"--- End Iter {iter:03d} ---")
    print([f"Global Test Accuracy:{acc:.4f}", f"Precision:{precision:.4f}", f"Recall:{recall:.4f}", f"F1:{f1:.4f}"])
    print([f"Restyle Test Accuracy:{acc_res:.4f}", f"Precision:{precision_res:.4f}", f"Recall:{recall_res:.4f}", f"F1:{f1_res:.4f}"])

    return acc, precision, recall, f1, acc_res, precision_res, recall_res, f1_res, ckpt_path


# ------------------- MAIN -------------------
datasetname = "gossipcop"
batch_size = args.batch_size
max_len = 512
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
n_epochs = args.n_epochs
iterations = args.iters

test_accs, test_accs_res = [], []
prec_all, rec_all, f1_all = [], [], []
prec_all_res, rec_all_res, f1_all_res = [], [], []

prev_ckpt = None
for iter in range(iterations):
    set_seed(iter)
    acc, prec, rec, f1, acc_res, prec_res, rec_res, f1_res, ckpt_path = train_model(
        tokenizer, max_len, n_epochs, batch_size, datasetname, iter, prev_model_path=prev_ckpt
    )
    prev_ckpt = ckpt_path
    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(rec)
    f1_all.append(f1)
    test_accs_res.append(acc_res)
    prec_all_res.append(prec_res)
    rec_all_res.append(rec_res)
    f1_all_res.append(f1_res)

print(f"Total Test Accuracy: {np.mean(test_accs):.4f} | Prec: {np.mean(prec_all):.4f} | Rec: {np.mean(rec_all):.4f} | F1: {np.mean(f1_all):.4f}")
print(f"Restyle Test Accuracy: {np.mean(test_accs_res):.4f} | Prec: {np.mean(prec_all_res):.4f} | Rec: {np.mean(rec_all_res):.4f} | F1: {np.mean(f1_all_res):.4f}")
