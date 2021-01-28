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


class TransformersClassification:

    def __init__(self, config):
        # self.name, self.num_classes, epochs, batchs
        
        self.Configs = config
        self.num_classes = len(config.label_list)
        
        self.train_logits = []
        self.validation_logits = []
        self.test_logits = []

        self.train_texts = []
        self.train_labels = []
        self.validation_texts = []
        self.validation_labels = []
        self.test_texts = []
        self.test_labels = []

        train = pd.read_csv(os.path.join(self.Configs.data_dir, 'train.csv'))
        
        try:
            dev = pd.read_csv(os.path.join(self.Configs.data_dir, 'dev.csv'))
        
        except:
            print('Validation disabled.')
        test = pd.read_csv(os.path.join(self.Configs.data_dir, 'test.csv'))
        
        self.train_texts = train['text'].tolist()
        
        self.train_labels = train['label'].tolist()
        
        try:
            self.validation_texts = dev['text'].tolist()
            self.validation_labels = dev['label'].tolist()

        except:
            pass
        self.test_texts = test['text'].tolist()
        
        for i in range(len(self.test_texts)):
            self.test_labels.append(0)

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        
        if self.Configs.model_name == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(self.Configs.pretrained_model_dir,
                                                                         num_labels=self.num_classes)
            self.tokenizer = BertTokenizer.from_pretrained(self.Configs.pretrained_model_dir)
        
        if self.Configs.model_name == 'albert':
            self.model = AlbertForSequenceClassification.from_pretrained(self.Configs.pretrained_model_dir,
                                                                         num_labels=self.num_classes)
            self.tokenizer = AlbertTokenizer.from_pretrained(self.Configs.pretrained_model_dir)
        
        if self.Configs.model_name == 'distilbert':
            self.model = DistilBertForSequenceClassification.from_pretrained(self.Configs.pretrained_model_dir,
                                                                             num_labels=self.num_classes)
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.Configs.pretrained_model_dir)
           
        if self.Configs.model_name == 'roberta':
            self.model = RobertaForSequenceClassification.from_pretrained(self.Configs.pretrained_model_dir,
                                                                          num_labels=self.num_classes)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.Configs.pretrained_model_dir)
            
        if torch.cuda.is_available():
            self.model.cuda()
    

    def train(self):

        # Combine the training inputs into a TensorDataset.
        train_params = tokenize_sentences(self.train_texts,self.Configs.max_seq_length,self.tokenizer,self.train_labels)
        train_dataset = TensorDataset(train_params['input_ids'], train_params['attention_masks'],
                                      train_params['labels'])

        if self.Configs.do_eval == True:
            
            validation_params = tokenize_sentences(self.validation_texts, 
                                                       self.Configs.max_seq_length, self.tokenizer, self.validation_labels)
            validation_dataset = TensorDataset(validation_params['input_ids'], validation_params['attention_masks'],
                                               validation_params['labels'])

    
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=self.Configs.train_batch_size  # Trains with this batch size.
        )

        if self.Configs.do_eval == True:
            validation_dataloader = DataLoader(
                validation_dataset,  # The training samples.
                sampler=RandomSampler(validation_dataset),  # Select batches randomly
                batch_size=self.Configs.eval_batch_size # Trains with this batch size.
            )

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(self.model.parameters(),
                          lr=self.Configs.learning_rate,  # default is 5e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

 
        epochs = self.Configs.num_train_epochs

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=total_steps / 10,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

  
        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0
            train_logits = []
            validation_logits = []
            
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print(
                        '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                loss, logits = self.model(b_input_ids,

                                          attention_mask=b_input_mask,
                                          labels=b_labels)

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                train_logits.append(logits)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. 
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            torch.cuda.empty_cache()

            if self.Configs.do_eval == True:
                
                # After the completion of each training epoch, measure our performance on
                # our validation set.

                print("")
                print("Running Validation...")

                t0 = time.time()

                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                self.model.eval()

                # Tracking variables
                total_eval_accuracy = 0
                total_eval_loss = 0
                nb_eval_steps = 0

                # Evaluate data for one epoch
                for batch in validation_dataloader:
                    
                    b_input_ids = batch[0].to(self.device)
                    b_input_mask = batch[1].to(self.device)
                    b_labels = batch[2].to(self.device)

                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    with torch.no_grad():
                        
                        loss, logits = self.model(b_input_ids,
                                                  attention_mask=b_input_mask,
                                                  labels=b_labels)

                        # Move logits and labels to CPU
                        logits = logits.detach().cpu().numpy()
                        label_ids = b_labels.to('cpu').numpy()
                        validation_logits.append(logits)
                    # Accumulate the validation loss.
                    # total_eval_loss += loss.item()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += flat_accuracy(logits, label_ids)

                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(validation_dataloader)

                # Measure how long the validation run took.
                validation_time = format_time(time.time() - t0)

                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print("  Validation took: {:}".format(validation_time))


            # torch.cuda.empty_cache()

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss if self.Configs.do_eval == True else 'NA',
                    'Valid. Accur.': avg_val_accuracy if self.Configs.do_eval == True else 'NA',
                    'Training Time': training_time,
                    'Validation Time': validation_time if self.Configs.do_eval == True else 'NA'
                }
            )
        
        if self.Configs.do_eval == True:
            validation_logits = np.vstack(validation_logits)
        
        train_logits = np.vstack(train_logits)

        self.train_logits = train_logits
        self.validation_logits = validation_logits

        # torch.state_dict(self.model,'/content/checkpoint.pth')
        output_dir = self.Configs.output_dir

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


    def test(self):

        test_params = tokenize_sentences(self.test_texts,self.Configs.max_seq_length,self.tokenizer,self.test_labels)
        
        prediction_data = TensorDataset(test_params['input_ids'], test_params['attention_masks'], test_params['labels'])
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.Configs.predict_batch_size)

        
        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions = []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                loss, logits = self.model(b_input_ids,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.extend(logits)
         
        self.test_logits = predictions
        torch.cuda.empty_cache()

        return [scipy.special.expit(pred) for pred in predictions]
    
    
    def text_inference(self, texts):

        inf_labels = []
        for i in range(len(texts)):
            inf_labels.append(0)

        test_params = tokenize_sentences(texts, self.Configs.max_seq_length, self.tokenizer, inf_labels)
        prediction_data = TensorDataset(test_params['input_ids'], test_params['attention_masks'],
                                        test_params['labels'])
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                           batch_size=self.Configs.predict_batch_size)

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
           
        self.test_logits = predictions
        torch.cuda.empty_cache()
        return [scipy.special.expit(pred) for pred in predictions]


def add_CLS_and_SEP(sentences,tokenizer):
    max_len = 0

    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    return sentences


def tokenize_sentences(sentences,max_seq_len,tokenizer,labels = []):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    sentences = add_CLS_and_SEP(sentences,tokenizer)

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



# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)




def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))