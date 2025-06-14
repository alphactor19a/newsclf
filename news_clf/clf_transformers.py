import os
import os.path as op

import torch
from torch import nn
from torch import logit
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoModelForSequenceClassification,AutoTokenizer
from safetensors import safe_open

def load_checkpoint(checkpoint):
    if checkpoint.endswith('.safetensors'):
        tensors = {}
        with safe_open(checkpoint, framework='pt', device='cpu') as file:
            for k in file.keys():
                tensors[k] = file.get_tensor(k)
    else:
        tensors = torch.load(checkpoint)
    return tensors

from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForSequenceClassification

from .text_utils import shorten_to_n_words, format_prompt_with_article, format_prompt_from_row

def _class_probabilities(cumulative_probabilities):
    P = cumulative_probabilities
    K = P.shape[-1]+1
    result = []
    for k in range(K):
        if k == 0:
            result.append( P[:,k].unsqueeze(1) )
        elif k < K-1:
            result.append( (P[:,k] - P[:,k-1]).unsqueeze(1) )
        else:
            result.append( (1 - P[:,k-1]).unsqueeze(1) )
    
    result = torch.cat(result, dim=-1)
    return result

def _predict_class(cumulative_probabilities):
    class_probabilities = _class_probabilities(cumulative_probabilities)
    return class_probabilities.argmax(dim=-1)

# define ordinal classification head
class OrdinalRegressionHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, link_function=nn.Sigmoid(), 
                 dtype=torch.float32, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        
        thresh_init = torch.tensor([0]+[1]*(num_classes-2), dtype=torch.float32)
        self.raw_thresholds = nn.Parameter(thresh_init, requires_grad=True)
        self.link_function = link_function

        if isinstance(link_function, nn.Sigmoid):
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.BCELoss()
        self.device = device
        self = self.to(device)
        
    @property
    def theta(self):
        return torch.cumsum(self.raw_thresholds**2, dim=0)
    
    def forward(self, x, targets=None, verbose=False):
        # x is the [CLS] hidden states
        logits = self.linear(x.to(self.raw_thresholds.dtype)).squeeze(-1)  # shape: [batch]
        thresholds = self.theta 
        logits = logits.unsqueeze(1)
        thresholds = thresholds.unsqueeze(0).repeat(logits.size(0), 1)
        
        batch_size = x.shape[0]
        
        threshold_logits = thresholds - logits
        probs = self.link_function(threshold_logits)
        
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                targets = torch.LongTensor(targets)
            
            targets = targets.to(x.device).unsqueeze(-1)
            range_ = torch.arange(self.num_classes-1).unsqueeze(0)
            range_ = range_.repeat_interleave(batch_size, 0).to(x.device)

            bce_targets = (targets <= range_).to(x.dtype)
            
            if verbose:
                print('targets', targets)
                #print('range', range_)
                print('bce_targets', bce_targets)
                print('logits', logits)
                print('cum_probs', probs)
                print('class probabilities', _class_probabilities(probs))
                print('theta', self.theta)
                print(self.link_function, self.loss_func)
            
            if isinstance(self.link_function, nn.Sigmoid):
                # use BCEWithLogitsLoss for numerical stability
                loss = self.loss_func(threshold_logits, bce_targets)
            else:
                loss = self.loss_func(probs, bce_targets)
        else:
            loss = None
        
        return threshold_logits, probs, loss, logits

class PretrainedModelForOrdinalSequenceClassification(nn.Module):
    def __init__(self, model_id='microsoft/deberta-v3-xsmall', 
                       num_classes=3, link_function=nn.Sigmoid(), 
                       device_map='auto', checkpoint=None,
                       class_labels=['Neutral', 'Loaded', 'Alarmist']):
        super(PretrainedModelForOrdinalSequenceClassification, self).__init__()
        self.model = AutoModel.from_pretrained(model_id, 
                                   attn_implementation='eager', # flash & sdpa not supported for deberta
                                   torch_dtype=torch.float32,
                                   device_map=device_map,
                                   )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.class_labels = class_labels
        assert len(class_labels) == num_classes, 'Number of labels (`class_labels`) ' + \
                                                 'must match the number of classes'
        
        self.device = self.model.device
        self.num_classes = num_classes
        self.hidden_dim = self.model.config.hidden_size
        self.clf_head = OrdinalRegressionHead(self.hidden_dim, 
                                              num_classes, 
                                              link_function=link_function,
                                              dtype=torch.float32,
                                              device=self.model.device)
        if checkpoint is not None:
            self.load_state_dict(load_checkpoint(checkpoint))
        
        self.device = self.model.device
    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.model.gradient_checkpointing_enable(*args, **kwargs)
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        targets = labels
        dev = self.model.device
        outputs = self.model(input_ids=input_ids.to(dev), 
                             attention_mask=attention_mask.to(dev), 
                             **kwargs)
        x = outputs.last_hidden_state[:,0,:] # [CLS] token embedding
        
        threshold_logits, probs, loss, logits = self.clf_head(x, targets=targets)
        
        clf_outputs = SequenceClassifierOutput(loss=loss, 
                                               logits=threshold_logits, 
                                               hidden_states=x, 
                                               attentions=outputs.attentions)
        class_probabilities = _class_probabilities(probs)
        class_predictions = _predict_class(probs)
        clf_outputs.class_probabilities = class_probabilities
        clf_outputs.predicted_class = class_predictions
        clf_outputs.article_score = logits
        return clf_outputs
    
    def classify_article(self, title, body, return_raw_outputs=False):
        article_input_text = format_prompt_with_article(title, body)
        # articles were truncated to 1500 tokens in training, so truncate to that length
        # to keep inputs in-distribution
        article_inputs = self.tokenizer([article_input_text], padding='longest', 
                                        truncation=True, max_length=1500,
                                        return_tensors='pt')
        article_inputs = { k: v.to(self.device) for k, v in article_inputs.items() }
        
        with torch.no_grad():
            article_outputs = self(**article_inputs)
        
        if return_raw_outputs:
            return article_outputs
        else:
            #return_items = { 'article_score': article_outputs.article_score, 
            #                 'class_probabilities': article_outputs.class_probabilities }
            predicted_class = int(article_outputs.class_probabilities.argmax(dim=-1).cpu()[0])
            cls_probability = float(article_outputs.class_probabilities.squeeze(0).cpu()[predicted_class])
            scale_score = float(article_outputs.article_score.squeeze(0).cpu())
            predicted_label = self.class_labels[predicted_class]
            return_string = f'{predicted_label} (P={cls_probability:.3f} with scale score {scale_score:.2f})'
            return return_string

class PretrainedModelForUnorderedSequenceClassification(nn.Module):
    # Nota Bene: Currently assumes a deberta model in initialization
    def __init__(self, model_id='microsoft/deberta-v3-xsmall', 
                       num_classes=3, device_map='auto', 
                       checkpoint=None, 
                       class_labels=['Neutral', 'Loaded', 'Alarmist']):
        super(PretrainedModelForUnorderedSequenceClassification, self).__init__()
        
        self.class_labels = class_labels
        assert len(class_labels) == num_classes, 'Number of labels (`class_labels`) ' + \
                                                 'must match the number of classes'
        
        clf_model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                                   attn_implementation='eager', # flash & sdpa not supported for deberta
                                   torch_dtype=torch.float32,
                                   device_map=device_map,
                                   num_labels=num_classes,
                                   )
        self.deberta = clf_model.deberta
        self.pooler = clf_model.pooler
        self.classifier = clf_model.classifier
        self.dropout = clf_model.dropout
        
        #self.clf_model = clf_model
        object.__setattr__(self, "clf_model", clf_model) # bypasses parameter registering via nn.Module, 
                                                         # else we get duplicate parameters
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.device = clf_model.device
        self.num_classes = num_classes
        self.hidden_dim = self.deberta.config.hidden_size
        
        if checkpoint is not None:
            self.load_state_dict(load_checkpoint(checkpoint))
        
        self.device = self.deberta.device
    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.model.gradient_checkpointing_enable(*args, **kwargs)
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
        model_inputs = { 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        model_inputs.update(kwargs)
        
        clf_outputs = self.clf_model(**model_inputs)
        return clf_outputs
    
    def classify_article(self, title, body, return_raw_outputs=False):
        article_input_text = format_prompt_with_article(title, body)
        # articles were truncated to 1500 tokens in training, so truncate to that length
        # to keep inputs in-distribution
        article_inputs = self.tokenizer([article_input_text], padding='longest', 
                                        truncation=True, max_length=1500,
                                        return_tensors='pt')
        article_inputs = { k: v.to(self.device) for k, v in article_inputs.items() }
        with torch.no_grad():
            article_outputs = self(**article_inputs)
        
        if return_raw_outputs:
            return article_outputs
        else:
            # compute the probabilities and return along with logits
            logits = article_outputs.logits
            probabilities = logits.softmax(axis=-1)
            #return_items = { 'classwise_logits': logits, 
            #                 'class_probabilities': probabilities }
            predicted_class = int(probabilities.argmax(dim=-1).cpu()[0])
            cls_probability = float(probabilities.squeeze(0).cpu()[predicted_class])
            predicted_label = self.class_labels[predicted_class]
            return_string = f'{predicted_label} ({cls_probability:.3f})'
            return return_string


