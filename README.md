# ClassiTransformers

[ClassiTransformers](https://pypi.org/project/classitransformers/) is an abstract library based on Tensorflow implementation of BERT and ELECTRA, and transformers library of HuggingFace Inc. 

Currently implemented models
- BERT (Tensorflow)
- ELECTRA (Tensorflow)
- RoBERTa (PyTorch - transformers library)
- ALBERT (PyTorch - transformers library)
- DistilBERT (PyTorch - transformers library)

Task
- Sequence Classification

## Features

- Works for N-class classification problem where N is any number of classes.
- Easy to use. Takes away all the complexity of writing tensorflow or pytorch codes for training and testing classification models.
- It provides an methods to easily train, test and create deployable models in .pb and .bin format in just 5 steps.
- Hyperparameters can be easily modified without having to change the source code.
- Deployment ready module available.

# Table of contents

<!--ts-->
- [ClassiTransformers](#classi-transformers)
  - [Features](#features)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Structure](#structure)
  - [Quick Start](#quick-start)
  - [Inferencing post Finetuning](#inferencing-post-fine-tuning)
  - [Data Preparation](#data-preparation)
  - [Setting Configuration](#setting-configuration)
  - [Class Methods](#class-methods)
  - [Getting Language Models](#getting-language-models)
  - [Real Dataset Examples](#real-dataset-examples)
  - [Support and Contributions](#support-and-contributions)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)
 
# Installation
Assuming that anaconda environment is already installed,

- ClassiTransformer can be installed from PyPI using 

```
pip install classitransformers
```

- with requirements.txt

```
pip install -r requirements.txt
```

- with yml file, create conda environment

```
conda env create -f environment.yml
source activate env
```

# Usage

Example notebooks can be found in the `sample_notebooks` directory.

## Structure

* `classitransformers.pytransformers` - Includes all pytorch-based text classification models from transformers library.
* `classitransformers.tfelectra` - Includes tensorflow-based Electra model for text classification
* `classitransformers.tfbert` - Includes tensorflow-based BERT model for text classification
* `classitransformers.report` - Used for reporting performance metrics. (precision, recall, F1, confusion matric)
* `classitransformers.configs` - Used for initializing the hyperparameters of the language models. Also checkas and creates the necessary directories.
* `classitransformers.downloader` - Used for downloading any of the 5 language models.
* `classitransformers.inference` - Used for inferencing in production, Only requires fine-tuned model directory and few hyperparameters.

  
## Quick Start
Supports text classification with any number of labels.

```python

from classitransformers.report import metrics
from classitransformers.configs import Configs
from classitransformers.pytransformers import TransformersClassification

config = Configs(pretrained_model_dir = './models/DistilBert/',
              model_name='distilbert',
              num_train_epochs=3, 
              train_batch_size = 16,
              eval_batch_size = 8, 
              do_train = True, 
              do_eval = True, 
              label_list = ["0", "1", "2"],
              max_seq_length = 64,
              data_dir = "./datasets/financial_phrasebank/", 
              output_dir = "./transformers_output_sentiment/")

model = TransformersClassification(config) 

model.train()
prediction = model.test()

y_pred = [np.argmax(tup) for tup in prediction]
print(y_pred)
```

## Inferencing post Fine-tuning
Supports pytorch models to be directly used in deployment/production.

```python
from classitransformers.inference import InferenceModel

model = InferenceModel(num_classes = 3,
                       max_seq_length = 64,
                       batch_size = 8,
                       model_name='distilbert',
                       classification_model_dir = './transformers_output_sentiment/')

model.text_inference(['Market gives the best returns among all the options of investemnts'])
```

## Data Preparation

The **directory** for input files needs to be **specified** while creating the **config** object. The files should be named **train.csv**, **dev.csv**, **test.csv**. **test.csv** may or may not have the labels. Labels would be useful for generating the report after testing.

Please check `data_preparation_format.txt` for details.

## Setting Configuration

All the Hyperparameters are listed in the Configs class, and have standard default values. The values can be changed by modifying the parameters passed in the Configs constructor for config object.

```python
class Configs(object):
  """Fine-tuning hyperparameters."""

    def __init__(self, pretrained_model_dir = './Albert',
              data_dir = "./data/", output_dir = "./albert_output/",
              export_dir = None, model_name="albert", 
              model_size ="base", learning_rate = 5e-5, 
              num_train_epochs=3.0, train_batch_size = 16,
              eval_batch_size = 8, predict_batch_size = 8, do_train = True,
              do_eval = True, do_predict = False, label_list = ["0", "1"],
              do_lower_case = True, max_seq_length = 256, use_tpu = False,
              iterations_per_loop = 1000, save_checkpoint_steps = 1000000,
              warmup_proportion = 0.1, export_path ='./exported_bert_model')
```

These are the parameters to be specified for creating the config object of Configs class.

Args:
* `pretrained_model_dir` : The path for pretrained directory.
* `data_dir` : The path of the directory for the train,dev and test files.
* `output_dir` (optional): The directory where the fine-tuned model will be saved. If not given, model will be saved in the current directory.(checkpoint for TF, .bin for pytorch)
* `export_dir` (optional): The directory where the model to be deployed will be saved.(Currently only for BERT)
* `model_name` : The name of the model. Either of these: 'albert', 'bert', 'electra', 'roberta', 'distilbert'
* `learning_rate`: The learning rate required while training the model. Default is 5e-5.
* `num_training_epochs`: The number of iterations for finetuning the pretrained model for classification task.
* `label_list`: The list of the labels for text classification task.
* `max_seq_length`: Max Sequence Length (multiples of 2) should be ideally just greater than the length of the longest text sentence, to prevent loss of information.
* `export_path`: The export path directory where chkpt format is converted to .pb format. Only set for bert.


## Class Methods

The class methods do not take any parameters. All the parameters are predefined to improve the clarity of the code.

**`train()`**
Fine-Tunes(trains) the model and saves the model and config file in the `output_dir` directory. Validation is done after each epoch.

**`test()`**
Tests the model for test dataset. Returns the prediction labels.

**`export_model()`**
Exports checkpoint model to .pb fotmat. Used for tensorflow-serviing while inferencing.(Currently only for BERT)

**`inference()`**
Inference on any input csv in batches using tensorflow serving for .pb model. (Currently only for BERT)

**`text_inference()`**
Inference on list of sentences as input.

**`metrics()`**
Prints and returns the accuracy and other metrics. Also prints Confusion Matrix (decorated matrix using matplotlib)

**`downloader()`**
Downloads pre-trained models just by passing model names in lowercase.

## Getting Language Models

```python
from classitransformers.downloader import downloader

# pass name of the model ('albert', 'bert', 'electra', 'roberta', 'distilbert')
downloader('roberta') # Downloads to default dir '../models'
```

## Real Dataset Examples

* [BBC News and BBC Sports](http://mlg.ucd.ie/datasets/bbc.html)
* [Financial Phrasebank - 3 class Classification of Financial Statements](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news/kernels)
* [Semeval 2010 Task 8 - Entity Relationship Classification](https://www.aclweb.org/anthology/S10-1006.pdf)
* [Yelp 2013 Dataset - User Rating Classification](https://www.kaggle.com/c/yelp-recsys-2013/data)


## Support and Contributions

Please submit bug reports and feature requests as Issues.
Contributions are very welcome. 

For additional questions and feedback, please contact us at classitransformers@fmr.com

## Acknowledgement

ClassiTransformers is developed by Emerging Tech Team at Fidelity Investments.
The part of the package was developed as part of an internship program at Fidelity. We thank [Hrushikesh](https://github.com/Hrushikesh-Vazurkar) and [Mayank](https://github.com/mgroovyank) for their contribution to the package.

## License

ClassiTransformers is licensed under the [Apache License 2.0.](LICENSE.md)
