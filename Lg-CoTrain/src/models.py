import torch.nn as nn
from transformers import RobertaModel, BertModel, DebertaModel, AutoModel, DebertaConfig



class RoBERTa(nn.Module):
    def __init__(self, num_classes, args=None):
        super(RoBERTa, self).__init__()
        self.bert = RobertaModel.from_pretrained(
            "roberta-base", 
            output_attentions = False, 
            output_hidden_states = True, 
            return_dict=False
        )
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, num_classes)
        self.task = args.dataset
        if self.task in ['swag', 'hellaswag']:
            self.n_choices = 4
    def forward(self, input_ids, attention_mask,labels=None):
        if self.task in ['swag', 'hellaswag']:
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            
        _, pooled_output, _= self.bert(input_ids = input_ids, attention_mask = attention_mask)
       
        output = self.linear(pooled_output)
        if self.task in ['swag', 'hellaswag']:
            output = output.view(-1, n_choices)
        return output
    



class BERT(nn.Module):
    def __init__(self, num_classes,args=None):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=False, output_hidden_states=True, return_dict=False)
        self.linear = nn.Linear(768, num_classes)
        self.task = args.dataset
        if self.task in ['swag', 'hellaswag']:
            self.n_choices = 4
            
        
    def forward(self, input_ids, attention_mask, labels=None):
        if self.task in ['swag', 'hellaswag']:
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output, _= self.bert(input_ids = input_ids, attention_mask = attention_mask)
        output = self.linear(pooled_output)
        if self.task in ['swag', 'hellaswag']:
            output = output.view(-1, n_choices)
        return output



    
class RoBERTaLarge(nn.Module):
    def __init__(self, num_classes, args=None):
        super(RoBERTaLarge, self).__init__()
        self.bert = RobertaModel.from_pretrained("roberta-large", 
                                                output_attentions=False, 
                                                output_hidden_states=True, 
                                                return_dict=False)
        self.linear = nn.Linear(1024, num_classes)  # RoBERTa-large has 1024 hidden size
        self.task = args.dataset
        if self.task in ['swag', 'hellaswag']:
            self.n_choices = 4
            
    def forward(self, input_ids, attention_mask, labels=None):
        if self.task in ['swag', 'hellaswag']:
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            
        _, pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        
        if self.task in ['swag', 'hellaswag']:
            output = output.view(-1, n_choices)
        return output

class DeBERTa(nn.Module):
    def __init__(self, num_classes, args=None):
        super(DeBERTa, self).__init__()
        self.bert = DebertaModel.from_pretrained("microsoft/deberta-base", 
                                               output_attentions=False, 
                                               output_hidden_states=True, 
                                               return_dict=False)
        self.linear = nn.Linear(768, num_classes)  # DeBERTa-base has 768 hidden size
        self.task = args.dataset
        if self.task in ['swag', 'hellaswag']:
            self.n_choices = 4
            
    def forward(self, input_ids, attention_mask, labels=None):
        if self.task in ['swag', 'hellaswag']:
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            
        # Note: DeBERTa returns (last_hidden_state, hidden_states) when return_dict=False
        last_hidden_state, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # For DeBERTa, we take the first token ([CLS]) as the pooled output
        pooled_output = last_hidden_state[:, 0, :]
        output = self.linear(pooled_output)
        
        if self.task in ['swag', 'hellaswag']:
            output = output.view(-1, n_choices)
        return output
    
    
    
    

class TransformerModel(nn.Module):
    def __init__(self, model_name, num_classes, dataset=None):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(
            model_name,  # Dynamically specify the model name
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True  # Use return_dict for structured outputs
        )
        hidden_size = self.transformer.config.hidden_size
        self.linear = nn.Linear(hidden_size, num_classes)
        self.task = dataset
        if self.task in ['swag', 'hellaswag']:
            self.n_choices = 4

    def forward(self, input_ids, attention_mask, labels=None):
        if self.task in ['swag', 'hellaswag']:
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        # Forward pass through the transformer model
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Access last hidden state
        cls_embedding = last_hidden_state[:, 0, :]  # Take [CLS] token embedding
        output = self.linear(cls_embedding)
        

        if self.task in ['swag', 'hellaswag']:
            output = output.view(-1, n_choices)
        return output
    
# class DeBERTa(nn.Module):
#     def __init__(self, num_classes, args=None, model_name="deberta-base"):
#         super(DeBERTa, self).__init__()
#         # Validate model size
#         if model_name not in ["deberta-base", "deberta-large"]:
#             raise ValueError("model_size must be either 'deberta-base' or 'deberta-large'")
            
#         # Load the appropriate pretrained model
#         model_name = f"microsoft/{model_name}"
#         self.bert = DebertaModel.from_pretrained(
#             model_name,
#             output_attentions=False,
#             output_hidden_states=True,
#             return_dict=True
#         )
        
#         hidden_size = self.bert.config.hidden_size
#         self.linear = nn.Linear(hidden_size, num_classes)
#         self.task = args.dataset if args is not None else None
#         self.model_name = model_name
        
#         if self.task in ['swag', 'hellaswag']:
#             self.n_choices = 4

#     def forward(self, input_ids, attention_mask, labels=None):
#         if self.task in ['swag', 'hellaswag']:
#             n_choices = input_ids.size(1)
#             input_ids = input_ids.view(-1, input_ids.size(-1))
#             attention_mask = attention_mask.view(-1, attention_mask.size(-1))

#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs.last_hidden_state
#         cls_embedding = last_hidden_state[:, 0, :]  # [CLS] token
#         output = self.linear(cls_embedding)

#         if self.task in ['swag', 'hellaswag']:
#             output = output.view(-1, n_choices)
#         return output