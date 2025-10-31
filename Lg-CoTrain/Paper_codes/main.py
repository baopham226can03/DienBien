from comet_ml import Experiment
import os
import sys
import gc
import random
import time
import torch
import argparse
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    BertTokenizer,
    get_scheduler,
    logging as transformers_logging
)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import logging as lg

from dotenv import load_dotenv
load_dotenv()

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import delete_saved_models, log_message, str2bool
from src.data_processor import TextDataset
from src.models import RoBERTa, BERT, DeBERTa, RoBERTaLarge, TransformerModel
from src.loss import SmoothCrossEntropyLoss
# from gen_init_weights import WeightGenerator
# from co_training_parallel import CoTrainer
# from fine_tune_models import DualModelTrainer
from trainer_classes import WeightGenerator, CoTrainer, DualModelTrainer

# Constants
ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 300
EPOCH_PATIENCE = 5

# Dataset configurations
LABELED_SAMPLES = {
    'sci_nli': [1000, 1000],
    'ag_news': [40, 200],
    'yahoo_answers': [500, 2000],
    'yelp_review': [250, 1000],
    'amazon_review': [250, 1000],
    'aclImdb': [40, 100],
    'qqp': [100, 500],
    'swag': [200, 1000],
    'hellaswag': [200, 1000],
    'mnli': [150, 750]
}

NUM_CLASSES = {
    'sci_nli': 4,
    'ag_news': 4,
    'yahoo_answers': 10,
    'yelp_review': 5,
    'amazon_review': 5,
    'aclImdb': 2,
    'qqp': 2,
    'swag': 1,
    'hellaswag': 1,
    'mnli': 3
}

# Model mapping for easier reference
HF_MODEL_MAPPING = {
    "phi-3": "Phi-3-medium-4k",
    "phi-3-128k": "Phi-3-medium-128k",
    "mistral-7b": "Mistral-7B-Instruct",
    "llama-3-8b": "Llama-3.1-8B",
    "llama-3-70b": "Llama-3.3-70B",
    "roberta": "roberta-base"
}
PLM_ID_MAPPING = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "deberta-base": "microsoft/deberta-base",
    "deberta-large": "microsoft/deberta-large",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased"
}

few_shot_samples_per_class = {
    "ag_news": 4,
    "aclImdb": 2,
    "yahoo_answers": 3,
    "amazon_review": 2,
    "yelp_review": 2,
    "qqp": 1,
    "swag": 1,
    "hellaswag": 1,
    "mnli": 1
}

plm_ids = list(PLM_ID_MAPPING.keys())
llm_ids = list(HF_MODEL_MAPPING.keys())
datasets = list(LABELED_SAMPLES.keys())




# python3 main.py --dataset yelp_review --labeled_sample_idx 0 --hf_model_id_short phi-3 --seed 1234 --plm_id roberta-base --few_shot --cuda_devices 0,1
def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Co-Training Script")
    parser.add_argument("--dataset", type=str,  choices=datasets, help="Dataset name")
    parser.add_argument("--labeled_sample_idx", type=int, choices=[0, 1], help="Index for labeled samples")
    parser.add_argument("--hf_model_id_short", type=str, choices=llm_ids, help="Short ID for the Hugging Face model")
    parser.add_argument("--seed", type=int, default=1234, choices=[1234, 4567, 8998], help="Random seed for reproducibility")
    parser.add_argument("--plm_id", type=str, default="roberta-base", choices=plm_ids, help="PLM (bert-base, roberta-base, deberta-base, etc.)")
    parser.add_argument("--pseudo_label_shot", type=int, default=0, help="Number of pseudo labeled samples")
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot prompted pseudolabels.")
    parser.add_argument("--single_set", action="store_true", default=False, help="Use single training set for both models")
    parser.add_argument("--no_co_training", action="store_true", default=False, help="Disable co-training")
    parser.add_argument("--metric_combination", type=str, default='cv', choices=["cv", "cc"], help="Metric combination method")
    parser.add_argument("--exp_name", type=str, default="lg-cotr", help="Experiment name")
    parser.add_argument("--setup_local_logging", type=bool, default=False, help="Setup local logging")
    parser.add_argument("--comet_ml", action="store_true", default=False, help="Use comet_ml for experiment tracking")
    parser.add_argument("--use_correct_labels_only", type=str2bool, default=False, help="Use correct labels only")
    parser.add_argument("--cuda_devices", type=str, default="0,1", help="Comma-separated list of CUDA device IDs to use (e.g., 0,1)")
    parser.add_argument("--imb_training", action="store_true", default=False, help="Use imbalanced training")
    args = parser.parse_args()
    
    
    args.pseudo_label_shot = few_shot_samples_per_class[args.dataset] if args.few_shot else 0
    
    return args


def set_environment(args):
    """Set environment variables and random seeds."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    transformers_logging.set_verbosity_error()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device configuration
    if torch.cuda.device_count() >= 2:
        device_1 = torch.device("cuda:0")
        device_2 = torch.device("cuda:1")
    else:
        device_1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_2 = device_1
        
    return device_1, device_2


def setup_local_logging(args):
    """Set up logging to file and console."""
    if not args.setup_local_logging:
        return None
    
    log_dir = f"{ROOT}/output/{args.dataset}/{args.exp_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output_log_path = os.path.join(log_dir, f"log_{args.saved_model_name_suffix}.txt")
    
    lg.basicConfig(
        filename=output_log_path,
        filemode='w',
        level=lg.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = lg.getLogger()
    return logger



def setup_comet_experiment(args):
    """Set up Comet ML experiment."""
    if not args.comet_ml:
        return None
    
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=args.exp_name,
        workspace="mezbaur-rahman"
    )
    experiment.set_name(f"{args.dataset}_{args.saved_model_name_suffix}")
    return experiment
    


def load_dataset_helper(dataset, N, pseudo_label_shot, processed_dir, data_dir, use_correct_labels_only=None, mnli_split=None):
    """Helper function to load datasets based on dataset type."""
    
    def json2pd(filepath):
        return pd.read_json(filepath, orient='index')
    
    def load_data(file_name):
        return pd.read_csv(os.path.join(data_dir, dataset, file_name), sep='\t')
    
    if dataset == 'sci_nli':
        trainingSet_1 = load_data('train_1.tsv')
        trainingSet_2 = load_data('train_2.tsv')
        testingSet = load_data('test.tsv')
        validationSet = load_data('dev.tsv')
        llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
        auto_labeled_data = llm_labeled_traininSet.copy()
        auto_labeled_data['label'] = auto_labeled_data['gen_label']
        auto_labeled_data['id'] = auto_labeled_data.index
        # auto_labeled_data = auto_labeled_data[['id', 'sentence1', 'sentence2', 'label']]
    else:
        # For datasets using similar loading pattern
        if dataset == 'mnli':
            testingSet = json2pd(os.path.join(data_dir, dataset, f'test_{mnli_split}.json'))
            validationSet = json2pd(os.path.join(data_dir, dataset, f'validation_{mnli_split}.json'))
        else:
            testingSet = json2pd(os.path.join(data_dir, dataset, 'test.json'))
            validationSet = json2pd(os.path.join(data_dir, dataset, 'dev.json'))
            
        # Load IDs for train splits and auto-labeled data
        train_1_ids = np.load(os.path.join(data_dir, dataset, 'labeled_idx', f'N_{N}', 'train_1_ids.npy'))
        train_2_ids = np.load(os.path.join(data_dir, dataset, 'labeled_idx', f'N_{N}', 'train_2_ids.npy'))
        auto_labeled_data_ids = np.load(os.path.join(data_dir, dataset, 'labeled_idx', f'N_{N}', 'auto_labeled_data_ids.npy'))
        llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
            
        # Filter datasets by IDs
        trainingSet_1 = llm_labeled_traininSet[llm_labeled_traininSet['id'].isin(train_1_ids)].copy()
        trainingSet_2 = llm_labeled_traininSet[llm_labeled_traininSet['id'].isin(train_2_ids)].copy()
        auto_labeled_data = llm_labeled_traininSet[llm_labeled_traininSet['id'].isin(auto_labeled_data_ids)].copy()
        
        # Set labels appropriately
        trainingSet_1['label'] = trainingSet_1['ori_label']
        trainingSet_2['label'] = trainingSet_2['ori_label']
        auto_labeled_data['label'] = auto_labeled_data['gen_label']
    
    
    if use_correct_labels_only:
        auto_labeled_data = auto_labeled_data[auto_labeled_data['label'] == auto_labeled_data['ori_label']]
    return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data



import os
import numpy as np
import pandas as pd

# def load_imb_dataset_helper(dataset, N, pseudo_label_shot, processed_dir, data_dir, use_correct_labels_only=None, mnli_split=None):
#     """Helper function to load datasets with long-tailed label distribution."""

#     def json2pd(filepath):
#         return pd.read_json(filepath, orient='index')
    
#     def load_data(file_name):
#         return pd.read_csv(os.path.join(data_dir, dataset, file_name), sep='\t')
    
#     if dataset == 'sci_nli':
#         trainingSet_1 = load_data('train_1.tsv')
#         trainingSet_2 = load_data('train_2.tsv')
#         testingSet = load_data('test.tsv')
#         validationSet = load_data('dev.tsv')
#         llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
#         auto_labeled_data = llm_labeled_traininSet.copy()
#         auto_labeled_data['label'] = auto_labeled_data['gen_label']
#         auto_labeled_data['id'] = auto_labeled_data.index

#     else:
#         # Load test and validation sets
#         if dataset == 'mnli':
#             testingSet = json2pd(os.path.join(data_dir, dataset, f'test_{mnli_split}.json'))
#             validationSet = json2pd(os.path.join(data_dir, dataset, f'validation_{mnli_split}.json'))
#         else:
#             testingSet = json2pd(os.path.join(data_dir, dataset, 'test.json'))
#             validationSet = json2pd(os.path.join(data_dir, dataset, 'dev.json'))

#         # Load full LLM-labeled training set
#         llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
#         llm_labeled_traininSet = llm_labeled_traininSet.sample(frac=1.0, random_state=42).reset_index(drop=True)

#         # Get number of classes from ori_label
#         all_labels = llm_labeled_traininSet['ori_label'].unique()
#         num_classes = len(all_labels)

#         # Create long-tailed label distribution
#         long_tail_ratio = np.linspace(1.0, 0.1, num_classes)
#         np.random.shuffle(long_tail_ratio)  # shuffle to avoid fixed head/tail class order
#         long_tail_ratio = long_tail_ratio / long_tail_ratio.sum()
#         samples_per_class = (N * 2 * long_tail_ratio).astype(int)
#         samples_per_class = np.maximum(samples_per_class, 1)  # at least 1 per class

#         # Sample trainingSet_1 and trainingSet_2 from each class
#         training_1_list, training_2_list = [], []
#         used_ids = set()

#         for i, label in enumerate(all_labels):
#             class_df = llm_labeled_traininSet[llm_labeled_traininSet['ori_label'] == label]
#             n_samples = samples_per_class[i]
#             selected = class_df.head(n_samples)
#             split_point = n_samples // 2

#             training_1_list.append(selected.iloc[:split_point])
#             training_2_list.append(selected.iloc[split_point:])
#             used_ids.update(selected.index.tolist())

#         trainingSet_1 = pd.concat(training_1_list).copy()
#         trainingSet_2 = pd.concat(training_2_list).copy()
#         auto_labeled_data = llm_labeled_traininSet[~llm_labeled_traininSet.index.isin(used_ids)].copy()

#         # Set final labels
#         trainingSet_1['label'] = trainingSet_1['ori_label']
#         trainingSet_2['label'] = trainingSet_2['ori_label']
#         auto_labeled_data['label'] = auto_labeled_data['gen_label']
#         auto_labeled_data = auto_labeled_data[auto_labeled_data['label'] >= 0].copy()

#     # Optionally filter to only correct auto-labels
#     if use_correct_labels_only:
#         auto_labeled_data = auto_labeled_data[auto_labeled_data['label'] == auto_labeled_data['ori_label']]
    
#     return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data


def get_exponential_decay_ratio(num_classes, imbalance_ratio=10):
    cls_indices = np.arange(num_classes)
    ratios = imbalance_ratio ** (-cls_indices / (num_classes - 1))
    return ratios / ratios.sum()


def load_imb_dataset_helper(dataset, N, pseudo_label_shot, processed_dir, data_dir, use_correct_labels_only=None, mnli_split=None):
    """Helper function to load datasets with long-tailed label distribution."""

    def json2pd(filepath):
        return pd.read_json(filepath, orient='index')

    def load_data(file_name):
        return pd.read_csv(os.path.join(data_dir, dataset, file_name), sep='\t')

    if dataset == 'sci_nli':
        trainingSet_1 = load_data('train_1.tsv')
        trainingSet_2 = load_data('train_2.tsv')
        testingSet = load_data('test.tsv')
        validationSet = load_data('dev.tsv')
        llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
        auto_labeled_data = llm_labeled_traininSet.copy()
        auto_labeled_data['label'] = auto_labeled_data['gen_label']
        auto_labeled_data['id'] = auto_labeled_data.index

    else:
        # Load test and validation sets
        if dataset == 'mnli':
            testingSet = json2pd(os.path.join(data_dir, dataset, f'test_{mnli_split}.json'))
            validationSet = json2pd(os.path.join(data_dir, dataset, f'validation_{mnli_split}.json'))
        else:
            testingSet = json2pd(os.path.join(data_dir, dataset, 'test.json'))
            validationSet = json2pd(os.path.join(data_dir, dataset, 'dev.json'))

        # Load full LLM-labeled training set
        llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
        llm_labeled_traininSet = llm_labeled_traininSet.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Get number of classes from ori_label
        all_labels = llm_labeled_traininSet['ori_label'].unique()
        num_classes = len(all_labels)

        # Create long-tailed label distribution
        # long_tail_ratio = np.linspace(1.0, 0.1, num_classes)
        # long_tail_ratio = np.exp(np.linspace(np.log(1.0), np.log(0.01), num_classes))
        # ranks = np.arange(1, num_classes+1)
        # long_tail_ratio = 1/ranks  # Basic Zipf distribution
        
        
        
        # np.random.shuffle(long_tail_ratio)  # Shuffle to avoid fixed head/tail class order
        # long_tail_ratio = long_tail_ratio / long_tail_ratio.sum()
        long_tail_ratio = get_exponential_decay_ratio(num_classes, imbalance_ratio=10)
        print(f"Long tail ratio: {long_tail_ratio}")
        samples_per_class = (N * 2 * long_tail_ratio).astype(int)
        samples_per_class = np.maximum(samples_per_class, 1)

        # Sample trainingSet_1 and trainingSet_2 from each class
        training_1_list, training_2_list = [], []
        used_ids = set()

        for i, label in enumerate(all_labels):
            class_df = llm_labeled_traininSet[llm_labeled_traininSet['ori_label'] == label]
            n_samples = samples_per_class[i]
            selected = class_df.head(n_samples)
            split_point = n_samples // 2

            training_1_list.append(selected.iloc[:split_point])
            training_2_list.append(selected.iloc[split_point:])
            used_ids.update(selected.index.tolist())

        trainingSet_1 = pd.concat(training_1_list).copy()
        trainingSet_2 = pd.concat(training_2_list).copy()

        # Now subsample auto_labeled_data from remaining using same long_tail_ratio
        remaining = llm_labeled_traininSet[~llm_labeled_traininSet.index.isin(used_ids)].copy()
        remaining = remaining[remaining['gen_label'] >= 0]  # ensure valid gen labels

        auto_labeled_list = []
        for i, label in enumerate(all_labels):
            class_df = remaining[remaining['ori_label'] == label]
            class_n = int(len(remaining) * long_tail_ratio[i])
            selected = class_df.head(class_n)
            auto_labeled_list.append(selected)

        auto_labeled_data = pd.concat(auto_labeled_list).copy()
        auto_labeled_data['label'] = auto_labeled_data['gen_label']
        trainingSet_1['label'] = trainingSet_1['ori_label']
        trainingSet_2['label'] = trainingSet_2['ori_label']

    # Optionally filter to only correct auto-labels
    if use_correct_labels_only:
        auto_labeled_data = auto_labeled_data[auto_labeled_data['label'] == auto_labeled_data['ori_label']]

    return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data



def create_dataloader(dataframe, tokenizer, dataset_name, batch_size, max_len):
    """Create a DataLoader for a given dataset."""
    dataset_obj = TextDataset(dataframe, tokenizer, max_len, dataset=dataset_name)
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)




# def initialize_models(num_classes, args):
#     """Initialize models based on PLM type."""
#     if "roberta" in args.plm_id:
#         print(f"Using RoBERTa model: {args.plm_id}")
#         model_1 = RoBERTa(num_classes=num_classes, args=args)
#         model_2 = RoBERTa(num_classes=num_classes, args=args)
#     elif "bert" in args.plm_id:
#         print(f"Using BERT model: {args.plm_id}")
#         model_1 = BERT(num_classes=num_classes, args=args)
#         model_2 = BERT(num_classes=num_classes, args=args)
#     elif "deberta" in args.plm_id:
#         print(f"Using DeBERTa model: {args.plm_id}")
#         model_1 = DeBERTa(num_classes=num_classes, args=args)
#         model_2 = DeBERTa(num_classes=num_classes, args=args)
#     else:
#         print(f"Model type {args.plm_id} not recognized. Defaulting to RoBERTa base.")
#         model_1 = RoBERTa(num_classes=num_classes, args=args)
#         model_2 = RoBERTa(num_classes=num_classes, args=args)
#     return model_1, model_2

def initialize_models(num_classes, args):
    """Initialize models based on PLM type."""
    if "roberta-base" == args.plm_id:
        print(f"Using RoBERTa model: {args.plm_id}")
        model_1 = RoBERTa(num_classes=num_classes, args=args)
        model_2 = RoBERTa(num_classes=num_classes, args=args)
    elif "bert-base" == args.plm_id:
        print(f"Using BERT model: {args.plm_id}")
        model_1 = BERT(num_classes=num_classes, args=args)
        model_2 = BERT(num_classes=num_classes, args=args)
    elif "deberta-base" == args.plm_id:
        print(f"Using DeBERTa model: {args.plm_id}")
        model_1 = DeBERTa(num_classes=num_classes, args=args)
        model_2 = DeBERTa(num_classes=num_classes, args=args)
    elif "roberta-large" == args.plm_id:
        print(f"Using RoBERTa model: {args.plm_id}")
        model_1 = RoBERTaLarge(num_classes=num_classes, args=args)
        model_2 = RoBERTaLarge(num_classes=num_classes, args=args)
    else:
        print(f"Model type {args.plm_id} not recognized. Defaulting to RoBERTa base.")
        model_1 = RoBERTa(num_classes=num_classes, args=args)
        model_2 = RoBERTa(num_classes=num_classes, args=args)
    return model_1, model_2



def setup_optimization(model_1, model_2, dataloaders, training_params, criterion_class=nn.CrossEntropyLoss):
    """Set up optimizers, schedulers and criterion for training."""
    criterion = criterion_class(reduction='none')
    learning_rate = training_params['learning_rate']
    num_epochs = training_params['num_epochs']
    train_dataloader_1 = dataloaders['train_dataloader_1']
    train_dataloader_2 = dataloaders['train_dataloader_2']
    
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=learning_rate, weight_decay=0.01)
    
    num_training_steps_1 = num_epochs * len(train_dataloader_1)
    num_training_steps_2 = num_epochs * len(train_dataloader_2)
    
    lr_scheduler_1 = get_scheduler(
        name="linear", 
        optimizer=optimizer_1, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps_1
    )
    
    lr_scheduler_2 = get_scheduler(
        name="linear", 
        optimizer=optimizer_2, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps_2
    )
    
    optimizer_params = {
        'criterion': criterion,
        'optimizer_1': optimizer_1,
        'optimizer_2': optimizer_2,
        'num_training_steps_1': num_training_steps_1,
        'num_training_steps_2': num_training_steps_2,
        'lr_scheduler_1': lr_scheduler_1,
        'lr_scheduler_2': lr_scheduler_2
    }
    
    return optimizer_params

def get_batch_size(dataset, plm_id):
    if plm_id == 'bert-base':
        return 24 if dataset not in ['swag', 'hellaswag'] else 8
    elif plm_id == 'roberta-base':
        return 24 if dataset not in ['swag', 'hellaswag'] else 8
    elif plm_id == 'deberta-base':
        return 16 if dataset not in ['swag', 'hellaswag'] else 4
    else:
        # default fallback
        return 8



def evaluate_models(model_1, model_2, eval_dataloader, device_1, device_2):
    """Evaluate ensembled models on provided dataloader."""
    model_1.eval()
    model_2.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Process on first device
            batch_1 = {k: v.to(device_1) for k, v in batch.items()}
            outputs_1 = model_1(input_ids=batch_1['input_ids'], attention_mask=batch_1['attention_mask'])
            outputs_1 = outputs_1.logits if hasattr(outputs_1, 'logits') else outputs_1
            val_probs_1 = torch.nn.functional.softmax(outputs_1, dim=-1)
            
            # Process on second device
            batch_2 = {k: v.to(device_2) for k, v in batch.items()}
            outputs_2 = model_2(input_ids=batch_2['input_ids'], attention_mask=batch_2['attention_mask'])
            outputs_2 = outputs_2.logits if hasattr(outputs_2, 'logits') else outputs_2
            val_probs_2 = torch.nn.functional.softmax(outputs_2, dim=-1)
            
            # Ensemble predictions
            val_probs = val_probs_1.cpu() + val_probs_2.cpu()
            out_ensembled = torch.argmax(val_probs, dim=1)
            out_ensembled = out_ensembled.cpu().detach().numpy()
            
            # Collect predictions and ground truth
            y_pred_batch = out_ensembled.tolist()
            y_true_batch = batch_1['labels'].cpu().numpy().tolist()
            
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
    
    cur_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    return cur_f1, acc


def main():
    st = time.time()
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment and devices
    device_1, device_2 = set_environment(args)
    unique_devices = len(set([device_1, device_2]))

    
    
    # Determine model and dataset configurations
    # dataset = args.dataset
    N = LABELED_SAMPLES[args.dataset][args.labeled_sample_idx] // 2
    hf_model_name = HF_MODEL_MAPPING[args.hf_model_id_short]
    
    # Set pseudo_label_shot based on model
    if args.hf_model_id_short == "roberta":
        args.pseudo_label_shot = N * 2
    
    # Set up experiment name
    # args.exp_name = "lg-cotr"
    
    # Set up paths
    saved_model_name_suffix = f"_{args.exp_name}_{args.hf_model_id_short}_{args.pseudo_label_shot}_shot_{args.plm_id}_{N}_seed_{args.seed}"
    
    # Handle MNLI split if applicable
    mnli_split = 'matched' if args.dataset == 'mnli' else None
    if mnli_split:
        saved_model_name_suffix += f"_{mnli_split}"
        
    args.saved_model_name_suffix = saved_model_name_suffix
    
    # Set up directories
    data_dir = os.path.join(ROOT, 'data')
    saved_model_dir = f"{ROOT}/saved_models/{args.dataset}/{args.exp_name}"
    processed_dir = f"{ROOT}/processed/{args.dataset}/{args.hf_model_id_short}"
    # save_dir = os.path.join(processed_dir, f'N_{N}')
    
    args.saved_model_dir = saved_model_dir
    
    
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)
    
    # Set batch size based on dataset and args.plm_id
    BATCH_SIZE = get_batch_size(args.dataset, args.plm_id)
    
    
    # Set up hyperparameters
    hyper_params = {
        'BATCH_SIZE': BATCH_SIZE,
        'MAX_LEN': MAX_LEN,
        'EPOCH_PATIENCE': EPOCH_PATIENCE
    }
    
    # Set up logging and experiment tracking
    logger = setup_local_logging(args)
    comet_exp = setup_comet_experiment(args)
    args.logger = logger
    args.comet_exp = comet_exp
    
    log_message(message=f"Using devices: {device_1}, {device_2}", args=args)
    log_message(message=f"Devices: {device_1}, {device_2}", args=args)
    

    log_message(message=f'Starting log', args=args)
    log_message(message=f'Dataset: {args.dataset}, N: {N}, Seed: {args.seed}, HF Model: {hf_model_name}, NumShots: {args.pseudo_label_shot}, PLM: {args.plm_id}', args=args)
    
    
    # Load dataset
    
    if args.imb_training:
        trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data = load_imb_dataset_helper(
            args.dataset, N, args.pseudo_label_shot, processed_dir, data_dir,args.use_correct_labels_only, mnli_split
        )
    else:
        trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data = load_dataset_helper(
            args.dataset, N, args.pseudo_label_shot, processed_dir, data_dir,args.use_correct_labels_only, mnli_split
        )
    
    # print(len(trainingSet_1), len(trainingSet_2), len(testingSet), len(validationSet), len(auto_labeled_data))
    # time.sleep(1000)
    
    # # #make subsets of size X for each dataset 
    # X = 20
    # trainingSet_1 = trainingSet_1.sample(n=X, random_state=args.seed)
    # trainingSet_2 = trainingSet_2.sample(n=X, random_state=args.seed)
    # testingSet = testingSet.sample(n=X, random_state=args.seed)
    # validationSet = validationSet.sample(n=X, random_state=args.seed)
    # auto_labeled_data = auto_labeled_data.sample(n=X, random_state=args.seed)
    
    
    # If not using multiset, make both training sets the same
    if args.single_set:
        trainingSet_1 = pd.concat([trainingSet_1, trainingSet_2], ignore_index=True)
        trainingSet_2 = trainingSet_1.copy()
    
    # Initialize tokenizer
    if "roberta-base" == args.plm_id:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    elif "roberta-large" == args.plm_id:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=False)
    elif "bert-base" == args.plm_id:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif "deberta-base" == args.plm_id:
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base', do_lower_case=False)
    else:
        print(f"Tokenizer for {args.plm_id} not recognized. Defaulting to RoBERTa tokenizer.")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
        
        
    # tokenizer = AutoTokenizer.from_pretrained(PLM_ID_MAPPING[args.plm_id], do_lower_case=False)
    
    # Create dataloaders
    dataloaders = {
        'train_dataloader_1': create_dataloader(trainingSet_1, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'train_dataloader_2': create_dataloader(trainingSet_2, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'val_dataloader': create_dataloader(validationSet, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'test_dataloader': create_dataloader(testingSet, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'auto_label_dataloader': create_dataloader(auto_labeled_data, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN)
    }
    
    # Training parameters
    training_params = {
        'num_epochs': 10,
        'learning_rate': 2e-5,
        'accumulation_steps': int(64 / BATCH_SIZE)
    }
    
    # Initialize models and Set up optimizers and criterion for initial weight generation
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    optimizer_params = setup_optimization(model_1, model_2, dataloaders,training_params, criterion_class=nn.CrossEntropyLoss)
    
  
    
    
    # Generate initial weights
    log_message(message='Generating initial weights', args=args)
    generator = WeightGenerator(
        args=args,
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=(device_1, device_2),
        models=(model_1, model_2),
        auto_labeled_data=auto_labeled_data,
        # metric_combination='cv'
    )
    init_df = generator.generate_weights()
    
    # Re-initialize models for co-training and Set up optimizers with SmoothCrossEntropyLoss for co-training
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    optimizer_params = setup_optimization(model_1, model_2, dataloaders,training_params, criterion_class=SmoothCrossEntropyLoss)
    
    
    # Add init_df to dataloaders
    dataloaders['init_df_dataloader'] = create_dataloader(init_df, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN)
    
    
    # Co-training
    log_message(message='Starting co-training', args=args)
    trainer = CoTrainer(
        args=args,
        models={'model_1': model_1, 'model_2': model_2},
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=[device_1, device_2],
        init_df=init_df,
        # metric_combination='cv'
    )
    co_training_df = trainer.train()
    #save the co_training_df to a file
    co_training_df.to_csv(os.path.join(saved_model_dir, f'co_training_df{saved_model_name_suffix}.csv'), index=False)
    
    # print(co_training_df.columns)
    # print(co_training_df[['id', 'ori_label', 'gen_label','train_weights_1', 'train_weights_2', 'all_epoch_probabilities_1', 'all_epoch_probabilities_2']].head(10))
    # print(co_training_df.head(10))
    # time.sleep(5000)
    
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del model_1
    del model_2
    gc.collect()
    
    # Load co-trained models
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    model_1_path = f'{saved_model_dir}/co_trained_model_1{saved_model_name_suffix}.pt'
    model_2_path = f'{saved_model_dir}/co_trained_model_2{saved_model_name_suffix}.pt'
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    
    delete_saved_models(model_1_path)
    delete_saved_models(model_2_path)
    
    # Set up fine-tuning parameters
    training_params['num_epochs'] = 100
    hyper_params['EPOCH_PATIENCE'] = 10
    
    # Set up optimizers for fine-tuning
    optimizer_params = setup_optimization(
        model_1, model_2, 
        dataloaders,
        training_params
    )
    
    
    
    # Fine-tune models
    log_message(message='Fine-tuning models', args=args)
    dual_trainer = DualModelTrainer(
        args=args,
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=(device_1, device_2),
        models=(model_1, model_2)
    )
    dual_trainer.train()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del model_1
    del model_2
    gc.collect()
    
    # Load fine-tuned models
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    model_1_path = f'{saved_model_dir}/final_model_1{saved_model_name_suffix}.pt'
    model_2_path = f'{saved_model_dir}/final_model_2{saved_model_name_suffix}.pt'
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    
    delete_saved_models(model_1_path)
    delete_saved_models(model_2_path)
    
    model_1.to(device_1)
    model_2.to(device_2)
    
    # Evaluate models
    eval_split = 'val' if args.dataset in ['swag', 'hellaswag', 'qqp', 'mnli'] else 'test'
    eval_dataloader = dataloaders[f'{eval_split}_dataloader']
    
    cur_f1, acc = evaluate_models(model_1, model_2, eval_dataloader, device_1, device_2)
    
    # Log and print final results
    result_msg = (f"\n\nHf Model: {hf_model_name} PLM: {args.plm_id} Dataset: {args.dataset}, NumShots: {args.pseudo_label_shot}, "
                 f"N: {N} {eval_split.capitalize()} SEED: {args.seed} F1: {cur_f1:.4f}, "
                 f"{eval_split.capitalize()} Accuracy: {acc:.4f}")
    
    log_message(message=result_msg, args=args)
    
    msg = f"\nTotal time taken: {time.time() - st:.2f} seconds"
    log_message(message=msg, args=args)


if __name__ == "__main__":
    main()

# python3 main.py --dataset yelp_review --labeled_sample_idx 0 --hf_model_id_short phi-3 --seed 1234 --plm_id roberta-base --imb_training 
