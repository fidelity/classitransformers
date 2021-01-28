# Copyright 2020 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import os
import sys
import time
import torch
import scipy
import random
import logging
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer, BertTokenizer
from transformers import DistilBertForSequenceClassification, AlbertForSequenceClassification, RobertaForSequenceClassification, BertForSequenceClassification

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class InferenceModel:

    def __init__(self, num_classes, max_seq_length, batch_size, model_name, model_path):
        
        self.num_classes = num_classes
        self.classification_model_dir = model_path
        self.max_seq_length = max_seq_length
        self.predict_batch_size = batch_size
        self.model_name = model_name
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        
        if self.model_name == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(self.classification_model_dir,
                                                                         num_labels=self.num_classes)
            self.tokenizer = BertTokenizer.from_pretrained(self.classification_model_dir)
        
        if self.model_name == 'albert':
            self.model = AlbertForSequenceClassification.from_pretrained(self.classification_model_dir,
                                                                         num_labels=self.num_classes)
            self.tokenizer = AlbertTokenizer.from_pretrained(self.classification_model_dir)
        
        if self.model_name == 'distilbert':
            self.model = DistilBertForSequenceClassification.from_pretrained(self.classification_model_dir,
                                                                             num_labels=self.num_classes)
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.classification_model_dir)
           
        if self.model_name == 'roberta':
            self.model = RobertaForSequenceClassification.from_pretrained(self.classification_model_dir,
                                                                          num_labels=self.num_classes)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.classification_model_dir)
            
        if torch.cuda.is_available():
            self.model.cuda()

        
    
      
    def text_inference(self, texts):

        inf_labels = []
        for i in range(len(texts)):
            inf_labels.append(0)

        test_params = tokenize_sentences(texts, self.max_seq_length, self.tokenizer, inf_labels)
        prediction_data = TensorDataset(test_params['input_ids'], test_params['attention_masks'],
                                        test_params['labels'])
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                           batch_size=self.predict_batch_size)

        self.model.eval()
        predictions = []
        
        for batch in prediction_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                loss, logits = self.model(b_input_ids,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

            logits = logits.detach().cpu().numpy()
            predictions.extend(logits)
           
        torch.cuda.empty_cache()
        return [scipy.special.expit(pred) for pred in predictions]


def tokenize_sentences(sentences,max_seq_len,tokenizer,labels = []):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    for sent in sentences:
        
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    try:
        labels = torch.tensor(labels)
    except:
        labels = []
        for i in range(len(sentences)):
            labels.append[0]

    return {
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'labels': labels
    }
