# Copyright 2020 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

"""Set configurations for language models on a downstream classification task."""

import os

class Configs(object):
    
    """Fine-tuning hyperparameters."""


    """
    Parameters
    ----------
    
    pretrained_model_dir:   Directory path of pretrained models. (check downloader.py)
    
    model_name:             Name of the language model in small case (as string)
    
    learning_rate:          The initial learning rate for Adam.
    
    num_train_epochs:       Total number of training epochs to perform.
    
    train_batch_size:       Total batch size for training.
    
    eval_batch_size:        Total batch size for eval.
    
    predict_batch_size:     Total batch size for predict.
    
    do_train:               Whether to run training.
    
    do_eval:                Whether to run eval on the dev set.
    
    do_predict:             Whether to run training.
    
    label_list:             Number of classes in data
    
    do_lower_case:          Whether to lower case the input text. 
                            Should be True for uncased models and False for cased models
    
    max_seq_length:         The maximum total input sequence length after WordPiece tokenization. 
                            Sequences longer than this will be truncated, and shorter than this will be padded.
    
    data_dir:               The input data dir. Should contain the .csv files 
                            (train.csv, dev.csv, test.csv) for the task.
    
    output_dir:             The output directory where the model checkpoints will be written.
    
    num_tpu_cores:          Only used if `use_tpu` is True. Total number of TPU cores to use.
    
    gcp_project:            [Optional] Project name for the Cloud TPU-enabled project. 
                            If not specified, we will attempt to automatically detect the GCE project from metadata.
    
    master:                 [Optional] TensorFlow master URL.
    
    tpu_zone:               [Optional] GCE zone where the Cloud TPU is located in. 
                            If not specified, we will attempt to automatically detect the GCE project from metadata.
    
    tpu_name:               [Optional] GCE zone where the Cloud TPU is located in. 
                            If not specified, we will attempt to automatically detect the GCE project from metadata.
    
    use_tpu:                Whether to use TPU or GPU/CPU.
    
    iterations_per_loop:    How many steps to make in each estimator call.
    
    save_checkpoints_steps: How often to save the model checkpoint. 
    
    warmup_proportion:      Proportion of training to perform linear learning rate warmup for. 
                            E.g., 0.1 = 10% of training.
    
    export_dir:             The output model (.pb format) dir, where the freezed graph and weights get stored (only for BERT).
    
    tokenizer:              Wordpiece tokenizer object.
    
    export_path:            Path of saved .pb model for inferencing purpose (Only for BERT).
    
    """


    def __init__(self, pretrained_model_dir = './Albert',
              data_dir = "./data/",
              output_dir = "./albert_output/",
              export_dir = None,
              model_name="", model_size ="base",
              learning_rate = 5e-5, num_train_epochs=3.0, train_batch_size = 16,
              eval_batch_size = 8, predict_batch_size = 8, do_train = True,
              do_eval = True, do_predict = False, label_list = ["0", "1"],
              do_lower_case = True, max_seq_length = 256, use_tpu = False,
              iterations_per_loop = 1000, save_checkpoint_steps = 1000000,
              warmup_proportion = 0.1, export_path ='./exported_bert_model'):
        
        
        # default locations of required files
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.pretrained_model_dir = pretrained_model_dir
        
        # general
        if self.model_name == "":
            raise ValueError('Kindly specify the model name (bert, electra, albert, roberta, distilbert)')
        
        if not os.path.exists(self.data_dir):
                raise ValueError(self.data_dir, ' not found !!')
        
        if not os.path.exists(self.pretrained_model_dir):
            raise ValueError(pretrained_model_dir, ' pretrained model not found !!')
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        # Common paramteres for all models
        
        
        self.do_train = do_train  # train a model ?
        self.do_eval = do_eval  # evaluate the model ?
        self.do_predict = do_predict
        self.label_list = label_list
        
        self.num_train_epochs = num_train_epochs  # passes over the dataset during training
        self.warmup_proportion = warmup_proportion  # how much of training to warm up the LR for
        self.learning_rate = learning_rate
        self.do_lower_case = do_lower_case
        
        # Params related to sizes, common to all
        
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        
        # model
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 30522  # number of tokens in the vocabulary

        # training
        self.weight_decay_rate = 0.01
        self.layerwise_lr_decay = 0.8      
        self.save_checkpoints_steps = save_checkpoint_steps
        self.iterations_per_loop = iterations_per_loop
        self.use_tfrecords_if_existing = False    

        # TPU settings: Not required, just default init (required for execution)
        self.use_tpu = False
        self.num_tpu_cores = 1
        self.tpu_job_name = None
        self.tpu_name = None  # cloud TPU to use for training
        self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
        self.gcp_project = None  # project name for the Cloud TPU-enabled project
        self.master = None

        self.vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")
        
        # Specific to BERT tensorflow model
        if self.model_name == 'bert':
            self.init_checkpoint = os.path.join(pretrained_model_dir,'bert_model.ckpt')
            self.bert_config_file = os.path.join(pretrained_model_dir,'bert_config.json')
            if not os.path.exists(self.bert_config_file):
                raise ValueError('bert_config.json not found in pretrained_model_dir!!')
            if not os.path.exists(self.vocab_file):
                raise ValueError('vocab.txt not found in pretrained_model_dir!!')

            self.export_path = export_path
            self.export_dir = export_dir

            if not os.path.exists(self.export_path):
                os.makedirs(self.export_path)
            
        # Specific to Electra Tensorflow model
        elif self.model_name == 'electra':
            self.init_checkpoint = pretrained_model_dir
            self.model_size = model_size  # one of "small", "base", or "large"
            self.task_names = ["textclassification"]  # which tasks to learn
            self.model_hparam_overrides = {}
            self.num_trials = 1  # how many train+eval runs to perform
            self.keep_all_models = True
            self.log_examples = False

            if not os.path.exists(self.vocab_file):
                raise ValueError('vocab.txt not found in pretrained_model_dir!!')

            # default hyperparameters for different model sizes
            if self.model_size == "large":
                self.learning_rate = 5e-5
                self.layerwise_lr_decay = 0.9
            elif self.model_size == "small":
                self.embedding_size = 128

            # update defaults with passed-in hyperparameters
            self.tasks = {
              "textclassification":{
                "type":"classification",
                "labels":label_list,
                "header":True,
                "text_column":1,
                "label_column":2
                }
            }
