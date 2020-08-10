# Copyright 2020 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import os
import wget
import shutil
from zipfile import ZipFile


def downloader(model, destination_path='../models'):
    
    """
    Function to download pre-trained models from hugginf face aws repository OR Google's storgage.

    Parameters
    ----------
    model : Pick one of these ['bert','electra','roberta','distilbert','albert']
    download_path : Directory whether the model has to be downloaded

    Returns
    -------

    """
    
    model_download = {
            "bert": 'Bert_base',
            "electra": 'Electra_base',
            "roberta": 'Roberta',
            "distilbert" : 'DistilBert',
            "albert": 'Albert'
    }
    
    if model not in model_download:
        print("Please pick model name from ['bert','electra','roberta','distilbert','albert']")
        return None

    output_dir = os.path.join(destination_path, model_download[model])
    print("Model gets downloaded here: ", output_dir)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if model == 'roberta':
        
        
        config_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json',
                                    os.path.join(output_dir,'config.json'))
        vocab_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json',
                                    os.path.join(output_dir,'vocab.json'))
        merges_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt',
                                    os.path.join(output_dir,'merges.txt'))
        model_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin',
                                   os.path.join(output_dir,'pytorch_model.bin'))
    
    if model == 'albert':
        config_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-config.json',
                                    os.path.join(output_dir,'config.json'))
        spiece_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model',
                                    os.path.join(output_dir,'spiece.model'))
        model_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-pytorch_model.bin',
                                   os.path.join(output_dir,'pytorch_model.bin'))
        
    if model == 'distilbert':
        config_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json',
                                    os.path.join(output_dir,'config.json'))
        vocab_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt',
                                    os.path.join(output_dir,'vocab.json'))
        model_file = wget.download('https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin',
                                   os.path.join(output_dir,'pytorch_model.bin'))
    
    
    if model == 'bert':
        zip_file = wget.download('https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip',
                                    os.path.join(output_dir,'uncased_L-12_H-768_A-12.zip'))
        
        with ZipFile(os.path.join(output_dir,'uncased_L-12_H-768_A-12.zip'), "r") as zip_ref:
            zip_ref.extractall(output_dir)

        os.remove(os.path.join(output_dir,'uncased_L-12_H-768_A-12.zip'))
        
    if model == 'electra':
        zip_file = wget.download('https://storage.googleapis.com/electra-data/electra_base.zip',
                                  os.path.join(output_dir,'electra_base.zip'))
        
        with ZipFile(os.path.join(output_dir,'electra_base.zip'), "r") as zip_ref:
            zip_ref.extractall(output_dir)
            
        files = os.listdir(os.path.join(output_dir,'electra_base'))
        for f in files:
            shutil.move(os.path.join(output_dir,'electra_base',f), output_dir)

        os.remove(os.path.join(output_dir,'electra_base.zip'))
        os.rmdir(os.path.join(output_dir,'electra_base'))