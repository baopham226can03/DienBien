import torch
from torch.utils.data import Dataset
import re


class BaseDatasetProcessor:
    def process_dataframe(self, dataframe):
        raise NotImplementedError("Subclasses should implement this method")

    def extract_int_from_string(self, s):
        if isinstance(s, int):
            return s
        if isinstance(s, float):
            # You can choose how to handle floats: either convert if whole number, or ignore
            return int(s)
        if isinstance(s, str):
            match = re.search(r'\d+', s)
            return int(match.group()) if match else None
        return None


class GenericLabelProcessor:
    def get_numeric_label(self, label, label_map):
        if isinstance(label, int) and label in label_map.values():
            return label
        elif isinstance(label, str) and label.isdigit():
            return int(label)
        return label_map.get(label, -1)

    def get_textual_label(self, label, idx_to_label):
        return idx_to_label.get(label, -1)


class SciNLIDatasetProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def process_dataframe(self, dataframe):
        label_map = {'contrasting': 0, 'reasoning': 1, 'entailing': 2, 'neutral': 3}
        dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, label_map))
        dataframe['id'] = dataframe['id'].apply(self.extract_int_from_string)
        return dataframe[dataframe['label'] >= 0][['id', 'sentence1', 'sentence2', 'label']]


class MNLIDatasetProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        self.label_map = {} if label_map is None else label_map
        
    def process_dataframe(self, dataframe):
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['id'].apply(self.extract_int_from_string)
        dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        dataframe['sentence1'] = dataframe['premise']
        dataframe['sentence2'] = dataframe['hypothesis']
        return dataframe[['id', 'sentence1', 'sentence2', 'label']]


class QqpDatasetProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        self.label_map = {} if label_map is None else label_map
        
    def process_dataframe(self, dataframe):
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['id'].apply(self.extract_int_from_string)
        dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        dataframe['sentence1'] = dataframe['question1']
        dataframe['sentence2'] = dataframe['question2']
        return dataframe[['id', 'sentence1', 'sentence2', 'label']]


class SwagDatasetProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        self.label_map = {} if label_map is None else label_map
        
    def process_dataframe(self, dataframe):
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['id'].apply(self.extract_int_from_string)
        dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        dataframe['context'] = dataframe['sent1']
        dataframe['start_ending'] = dataframe['sent2']
        return dataframe[['id', 'context', 'start_ending', 'ending0', 'ending1', 'ending2', 'ending3', 'label']]


class HellaSwagDatasetProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        self.label_map = {} if label_map is None else label_map
        
    def process_dataframe(self, dataframe):
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['id'].apply(self.extract_int_from_string)
        dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        dataframe['context'] = dataframe['ctx_a']
        dataframe['start_ending'] = dataframe['ctx_b']
        return dataframe[['id', 'context', 'start_ending', 'endings', 'label']]


class TextOnlyProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        self.label_map = {} if label_map is None else label_map

    def process_dataframe(self, dataframe):
        
        if 'ori' in dataframe.columns:
            dataframe.rename(columns={'ori': 'sentence'}, inplace=True)
        
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['id'].apply(self.extract_int_from_string)
        dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        dataframe = dataframe[dataframe['label'] >= 0]
        
        return_keys = ['id', 'sentence', 'label']
        if 'aug_0' in dataframe.columns:
            return_keys.append('aug_0')
            return_keys.append('aug_1')
        if 'ori_label' in dataframe.columns:
            dataframe['ori_label'] = dataframe['ori_label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
            return_keys.append('ori_label')
        return dataframe[return_keys]
            
        # if 'aug_0' in dataframe.columns and 'aug_1' in dataframe.columns:
        #     return dataframe[['id', 'sentence', 'label','aug_0', 'aug_1']]
        # if 'ori_label' in dataframe.columns:
        #     dataframe['ori_label'] = dataframe['ori_label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        #     return dataframe[['id', 'sentence', 'label', 'ori_label']]
        # return dataframe[['id', 'sentence', 'label']]


# class TextDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len, dataset='sci_nli', model_type='roberta'):
#         self.dataset = dataset
#         self.encoder = TextEncoder(tokenizer, max_len)
#         self.dataframe = self.get_dataset_processor(dataset).process_dataframe(dataframe)

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]

#         if self.dataset in ['sci_nli', 'mnli', 'qqp']:
#             input_ids, token_type_ids, attention_mask = self.encoder.encode_pair_inputs(
#                 str(row['sentence1']), str(row['sentence2'])
#             )
#         elif self.dataset in ['ag_news', 'yahoo_answers', 'amazon_review', 'yelp_review', 'aclImdb']:
#             input_ids, token_type_ids, attention_mask = self.encoder.encode_sentence(
#                 str(row['sentence'])
#             )
#         elif self.dataset == 'swag':
#             endings = [str(row[f'ending{i}']) for i in range(4)]
#             input_ids, token_type_ids, attention_mask = self.encoder.encode_mc_inputs(
#                 str(row['context']), str(row['start_ending']), endings
#             )
#         elif self.dataset == 'hellaswag':
#             input_ids, token_type_ids, attention_mask = self.encoder.encode_mc_inputs(
#                 str(row['context']), str(row['start_ending']), row['endings']
#             )
#         else:
#             raise ValueError(f"Unsupported dataset: {self.dataset}")

#         item = {
#             'input_ids': input_ids,
#             'token_type_ids': token_type_ids,
#             'attention_mask': attention_mask,
#             'labels': torch.tensor(row['label'], dtype=torch.long),
#             'id': row['id']
#         }

#         return item

#     def get_dataset_processor(self, dataset):
#         label_maps = {
#             'ag_news': {'world': 0, 'sports': 1, 'business': 2, 'sci/tech': 3},
#             'yahoo_answers': {
#                 "Society & Culture": 0, "Science & Mathematics": 1, "Health": 2,
#                 "Education & Reference": 3, "Computers & Internet": 4, "Sports": 5,
#                 "Business & Finance": 6, "Entertainment & Music": 7,
#                 "Family & Relationships": 8, "Politics & Government": 9
#             },
#             'amazon_review': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
#             'yelp_review': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
#             'aclImdb': {'Positive': 0, 'Negative': 1},
#             'qqp': {'duplicate': 1, 'not_duplicate': 0},
#             'swag': {'0': 0, '1': 1, '2': 2, '3': 3},
#             'hellaswag': {'0': 0, '1': 1, '2': 2, '3': 3},
#             'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2}
#         }
#         processors = {
#             'sci_nli': SciNLIDatasetProcessor(),
#             'mnli': MNLIDatasetProcessor(label_maps['mnli']),
#             'qqp': QqpDatasetProcessor(label_maps['qqp']),
#             'swag': SwagDatasetProcessor(label_maps['swag']),
#             'hellaswag': HellaSwagDatasetProcessor(label_maps['hellaswag']),
#             'ag_news': TextOnlyProcessor(label_maps['ag_news']),
#             'yahoo_answers': TextOnlyProcessor(label_maps['yahoo_answers']),
#             'amazon_review': TextOnlyProcessor(label_maps['amazon_review']),
#             'yelp_review': TextOnlyProcessor(label_maps['yelp_review']),
#             'aclImdb': TextOnlyProcessor(label_maps['aclImdb'])
#         }
#         if dataset not in processors:
#             raise ValueError(f"Unsupported dataset: {dataset}")
#         return processors[dataset]

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, dataset='sci_nli', include_augmented=False):
        self.dataset = dataset
        self.encoder = TextEncoder(tokenizer, max_len)
        self.dataframe = self.get_dataset_processor(dataset).process_dataframe(dataframe)
        self.include_augmented = include_augmented
        # print(f'df columns: {self.dataframe.columns}')
        if self.include_augmented:
            if self.dataset not in ['ag_news', 'yahoo_answers', 'amazon_review', 'yelp_review', 'aclImdb']:
                raise ValueError(f"Augmented data is only available for ag_news, yahoo_answers, amazon_review, yelp_review, and aclImdb datasets")
            if 'aug_1' not in self.dataframe.columns:
                raise ValueError(f"Augmented data requested but 'aug_1' column not found in {dataset} dataframe")
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # print(row.keys())
        item = {
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'id': row['id']
        }
        if 'ori_label' in row:
            item['ori_labels'] = torch.tensor(row['ori_label'], dtype=torch.long)

        if self.dataset in ['sci_nli', 'mnli', 'qqp']:
            input_ids, token_type_ids, attention_mask = self.encoder.encode_pair_inputs(
                str(row['sentence1']), str(row['sentence2'])
            )
        elif self.dataset in ['ag_news', 'yahoo_answers', 'amazon_review', 'yelp_review', 'aclImdb']:
            input_ids, token_type_ids, attention_mask = self.encoder.encode_sentence(
                str(row['sentence'])
            )
            
            
            if self.include_augmented:
                # Weak augmentation (aug_0)
                aug0_input_ids, aug0_token_type_ids, aug0_attention_mask = self.encoder.encode_sentence(
                    str(row['aug_0'])
                )
                # Strong augmentation (aug_1)
                aug1_input_ids, aug1_token_type_ids, aug1_attention_mask = self.encoder.encode_sentence(
                    str(row['aug_1'])
                )
                item.update({
                    'aug0_input_ids': aug0_input_ids,
                    'aug0_token_type_ids': aug0_token_type_ids,
                    'aug0_attention_mask': aug0_attention_mask,
                    'aug1_input_ids': aug1_input_ids,
                    'aug1_token_type_ids': aug1_token_type_ids,
                    'aug1_attention_mask': aug1_attention_mask,
                })
        elif self.dataset == 'swag':
            endings = [str(row[f'ending{i}']) for i in range(4)]
            input_ids, token_type_ids, attention_mask = self.encoder.encode_mc_inputs(
                str(row['context']), str(row['start_ending']), endings
            )
        elif self.dataset == 'hellaswag':
            input_ids, token_type_ids, attention_mask = self.encoder.encode_mc_inputs(
                str(row['context']), str(row['start_ending']), row['endings']
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Add the common encoding fields
        item.update({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        })

        return item
    
    def get_dataset_processor(self, dataset):
        label_maps = {
            'ag_news': {'world': 0, 'sports': 1, 'business': 2, 'sci/tech': 3},
            'yahoo_answers': {
                "Society & Culture": 0, "Science & Mathematics": 1, "Health": 2,
                "Education & Reference": 3, "Computers & Internet": 4, "Sports": 5,
                "Business & Finance": 6, "Entertainment & Music": 7,
                "Family & Relationships": 8, "Politics & Government": 9
            },
            'amazon_review': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
            'yelp_review': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
            'aclImdb': {'Positive': 0, 'Negative': 1},
            'qqp': {'duplicate': 1, 'not_duplicate': 0},
            'swag': {'0': 0, '1': 1, '2': 2, '3': 3},
            'hellaswag': {'0': 0, '1': 1, '2': 2, '3': 3},
            'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        }
        processors = {
            'sci_nli': SciNLIDatasetProcessor(),
            'mnli': MNLIDatasetProcessor(label_maps['mnli']),
            'qqp': QqpDatasetProcessor(label_maps['qqp']),
            'swag': SwagDatasetProcessor(label_maps['swag']),
            'hellaswag': HellaSwagDatasetProcessor(label_maps['hellaswag']),
            'ag_news': TextOnlyProcessor(label_maps['ag_news']),
            'yahoo_answers': TextOnlyProcessor(label_maps['yahoo_answers']),
            'amazon_review': TextOnlyProcessor(label_maps['amazon_review']),
            'yelp_review': TextOnlyProcessor(label_maps['yelp_review']),
            'aclImdb': TextOnlyProcessor(label_maps['aclImdb'])
        }
        if dataset not in processors:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return processors[dataset]  


  
class TextEncoder:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_sentence(self, sentence):
        """Encode a single sentence and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_pair_inputs(self, sentence1, sentence2):
        """Encode a pair of sentences and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_mc_inputs(self, context, start_ending, endings):
        """Encode multiple choice inputs with context and multiple endings."""
        all_input_ids, all_token_type_ids, all_attention_masks = [], [], []
        
        for ending in endings:
            full_ending = f"{start_ending} {ending}" if start_ending else ending
            inputs = self.tokenizer.encode_plus(
                context,
                full_ending,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_tensors='pt'  # Return PyTorch tensors directly
            )
            
            all_input_ids.append(inputs['input_ids'].squeeze(0))
            all_token_type_ids.append(inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0))
            all_attention_masks.append(inputs['attention_mask'].squeeze(0))
            
        return torch.stack(all_input_ids), torch.stack(all_token_type_ids), torch.stack(all_attention_masks)