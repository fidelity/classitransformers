# Copyright 2020 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import os
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
import matplotlib.pyplot as plt
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from bert import modeling
from bert import optimization
from bert import tokenization

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class TextProcessor():
    """Processor for the data set."""

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a csv separated value file."""
        df = pd.read_csv(input_file).astype(str).values.tolist()
        return df

    def get_train_examples(self, data_dir):
        """read train.csv."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """read dev.csv."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """read test.csv."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """default_labels."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
           Assumes that,
           2nd column (index: 1) in test.csv is text input.
           2nd column (index: 1) in train.csv is text input.
           3rd column (index: 2) in train.csv is class label.
        """

        examples = []
        for (i, line) in enumerate(lines):
            # Train, dev and test set has a header
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:

                text_a = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[2])
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class BertClassification(object):
    
    """
    configs: CONFIGS object with configuration params
    """
    
    def __init__(self, configs = None):
        
        self.Configs = configs
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.Configs.vocab_file, do_lower_case=self.Configs.do_lower_case)
        
        bert_config = modeling.BertConfig.from_json_file(self.Configs.bert_config_file)
        
        if self.Configs.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (self.Configs.max_seq_length, bert_config.max_position_embeddings))

        tf.io.gfile.makedirs(self.Configs.output_dir)
        self.processor = TextProcessor()

        tpu_cluster_resolver = None
        if self.Configs.use_tpu and self.Configs.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            self.Configs.tpu_name, zone=self.Configs.tpu_zone, project=self.Configs.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.Configs.master,
            model_dir=self.Configs.output_dir,
            save_checkpoints_steps=self.Configs.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=self.Configs.iterations_per_loop,
              num_shards=self.Configs.num_tpu_cores,
              per_host_input_for_training=is_per_host))

        self.train_examples = None
        self.num_train_steps = None
        self.num_warmup_steps = None
        if self.Configs.do_train:
            self.train_examples = self.processor.get_train_examples(self.Configs.data_dir)
        self.num_train_steps = int(
            len(self.train_examples) / self.Configs.train_batch_size * self.Configs.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.Configs.warmup_proportion)

        model_fn = self.model_fn_builder(
          bert_config=bert_config,
          num_labels=len(self.Configs.label_list),
          init_checkpoint=self.Configs.init_checkpoint,
          learning_rate=self.Configs.learning_rate,
          num_train_steps=self.num_train_steps,
          num_warmup_steps=self.num_warmup_steps,
          use_tpu=self.Configs.use_tpu,
          use_one_hot_embeddings=self.Configs.use_tpu)

        self.estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=self.Configs.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=self.Configs.train_batch_size,
          eval_batch_size=self.Configs.eval_batch_size,
          predict_batch_size=self.Configs.predict_batch_size)

    def convert_single_example(self, example, label_list, max_seq_length,
                               tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)


        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        segment_ids = [0] * len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_id=label_id,
          is_real_example=True)
        return feature

    def convert_test_single_example(self, text, max_seq_length, tokenizer):
        
        """Converts a single test `InputExample` into a single `InputFeatures`."""

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        segment_ids = [0] * len(input_ids)

        feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_id=0
        )
        return feature

    def file_based_convert_examples_to_features(
        self, examples, label_list, max_seq_length, tokenizer, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""

        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(example, label_list,
                                         max_seq_length, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])
            features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def file_based_input_fn_builder(self, input_file, seq_length, is_training,
                                    drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        name_to_features = {
          "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "label_ids": tf.FixedLenFeature([], tf.int64),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
            return d

        return input_fn

    def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        
        model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)

    def model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_tpu,
                         use_one_hot_embeddings):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_real_example = None
            
            if "is_real_example" in features:
                is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = self.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                if use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                              init_string)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = optimization.create_optimizer(
                  total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  train_op=train_op,
                  scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.compat.v1.metrics.accuracy(
                        labels=label_ids, predictions=predictions, weights=is_real_example)
                    loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                    }

                eval_metrics = (metric_fn,
                              [per_example_loss, label_ids, logits, is_real_example])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  eval_metrics=eval_metrics,
                  scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  predictions={"probabilities": probabilities},
                  scaffold_fn=scaffold_fn)
            return output_spec

        return model_fn

    def train(self):
        """Finetunes the classifier according to the given training data."""

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        tokenization.validate_case_matches_checkpoint(self.Configs.do_lower_case,
                                                    self.Configs.init_checkpoint)

        if not self.Configs.do_train and not self.Configs.do_eval:
            raise ValueError(
            "At least one of `do_train` or `do_eval`.")

        if self.Configs.do_train:
            train_file = os.path.join(self.Configs.output_dir, "train.tf_record")
            self.file_based_convert_examples_to_features(
                self.train_examples, self.Configs.label_list, self.Configs.max_seq_length, 
                self.tokenizer, train_file)
            
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(self.train_examples))
            tf.logging.info("  Batch size = %d", self.Configs.train_batch_size)
            tf.logging.info("  Num steps = %d", self.num_train_steps)
            train_input_fn = self.file_based_input_fn_builder(
                input_file=train_file,
                seq_length=self.Configs.max_seq_length,
                is_training=True,
                drop_remainder=True)
            self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

        if self.Configs.do_eval:
            eval_examples = self.processor.get_dev_examples(self.Configs.data_dir)
            num_actual_eval_examples = len(eval_examples)

            eval_file = os.path.join(self.Configs.output_dir, "eval.tf_record")
            self.file_based_convert_examples_to_features(
                eval_examples, self.Configs.label_list, self.Configs.max_seq_length, 
                self.tokenizer, eval_file)

            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(eval_examples), num_actual_eval_examples,
                            len(eval_examples) - num_actual_eval_examples)
            tf.logging.info("  Batch size = %d", self.Configs.eval_batch_size)

            # This tells the estimator to run through the entire set.
            eval_steps = None

            eval_drop_remainder = True if self.Configs.use_tpu else False
            eval_input_fn = self.file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=self.Configs.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)

            result = self.estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            output_eval_file = os.path.join(self.Configs.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        
    def test(self, to_csv=True):
        """
        Predicts the model according to the given test data. 
        returns predictions of shape (n_samples, n_labels)
        """
        
        predict_examples = self.processor.get_test_examples(self.Configs.data_dir)
        num_actual_predict_examples = len(predict_examples)
        
        predict_file = os.path.join(self.Configs.output_dir, "predict.tf_record")
        self.file_based_convert_examples_to_features(predict_examples, self.Configs.label_list,
                                                self.Configs.max_seq_length, self.tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.Configs.predict_batch_size)

        predict_drop_remainder = True if self.Configs.use_tpu else False
        predict_input_fn = self.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.Configs.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
        
        result = self.estimator.predict(input_fn=predict_input_fn)
        
        predictions=[]
        for (i, prediction) in enumerate(result):
            predictions.append(prediction["probabilities"])
        print("Predictions = ",len(predictions))
        if to_csv:    
            output_predict_file = os.path.join(self.Configs.output_dir, "test_results.tsv")
            with tf.gfile.GFile(output_predict_file, "w") as writer:
                num_written_lines = 0
                tf.logging.info("***** Predict results *****saved in*****test_results.tsv")
                for (i, prediction) in enumerate(predictions):
                    probabilities = prediction
                    if i >= num_actual_predict_examples:
                        break
                    output_line = "\t".join(str(class_probability) for class_probability in probabilities) + "\n"
                    writer.write(output_line)
                    num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples
            
        return predictions

    def create_infer_model(self, input_ids, input_mask, segment_ids, labels, num_labels):
        """create inferencing model, returns probabilties"""

        model = modeling.BertModel(
          config=modeling.BertConfig.from_json_file(self.Configs.bert_config_file),
          is_training=False,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.sigmoid(logits)
        return probabilities

    
    def infer_fn_builder(self, num_labels):

        def model_fn(features):
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_real_example = None

            probabilities = self.create_infer_model(
                input_ids, input_mask, segment_ids, label_ids, num_labels)

            output_spec = tf.estimator.EstimatorSpec(
                predictions={"probabilities": probabilities},
                mode='infer'
            )
            return output_spec

        return model_fn

    
    def export_model(self):
        """exports a model checkpoint to .pb format (production/freezed graph and weigh)"""

        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
            input_ids = tf.placeholder(tf.int32, [None, self.Configs.max_seq_length], name='input_ids')
            input_mask = tf.placeholder(tf.int32, [None, self.Configs.max_seq_length], name='input_mask')
            segment_ids = tf.placeholder(tf.int32, [None, self.Configs.max_seq_length], name='segment_ids')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                'label_ids': label_ids,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            })()
            return input_fn

        model_fn = self.infer_fn_builder(num_labels=len(self.Configs.label_list))
        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           config=tf.estimator.RunConfig(model_dir=self.Configs.output_dir))

        estimator._export_to_tpu = False
        export_dir = estimator.export_saved_model(export_dir_base=self.Configs.export_path,
                                                  serving_input_receiver_fn=serving_input_fn)
        self.Configs.export_dir = export_dir.decode("utf-8")

        return export_dir.decode("utf-8")

    def chunks(self, text_list, size):
        """
        creates batches on text samples.
        
        Parameters
        ----------
        text_list : text samples of shape (n_samples,)
        size: batch size
        """
        
        for i in range(0, len(text_list), size):
            yield text_list[i:i+size]

    def inference(self, test_filename, batch_size=64):
        """
        performs inference using freezed (.pb) model.
        
        Parameters
        ----------
        test_filename : input file with format as of test.csv
        batch_size: Number of samples in a batch.
        """

        max_seq_length, tokenizer = self.Configs.max_seq_length, self.tokenizer
        df = pd.read_csv(test_filename)

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:

                tf.saved_model.loader.load(sess, [tag_constants.SERVING], self.Configs.export_dir)
                tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
                tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
                tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
                tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
                tensor_outputs = graph.get_tensor_by_name('Sigmoid:0')

                batches = list(self.chunks(df.text.tolist(),batch_size))

                predictions = []
                for i,batch in enumerate(batches):
                    print('Processing batch no. ',i)
                    features = [self.convert_test_single_example(txt, max_seq_length, tokenizer) for txt in batch]
                    input_ids = np.array([feature.input_ids for feature in features]).reshape(-1, max_seq_length)
                    input_mask = np.array([feature.input_mask for feature in features]).reshape(-1, max_seq_length)
                    label_ids = np.array([feature.label_id for feature in features])
                    segment_ids = np.array([feature.segment_ids for feature in features]).reshape(-1, max_seq_length)

                    pred = sess.run(tensor_outputs, feed_dict={
                            tensor_input_ids: input_ids,
                            tensor_input_mask: input_mask,
                            tensor_label_ids: label_ids,
                            tensor_segment_ids: segment_ids,
                        })
                    predictions.extend(pred)

        return predictions
    
    
    def text_inference(self, texts):

        max_seq_length, tokenizer = self.Configs.max_seq_length, self.tokenizer

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:

                tf.saved_model.loader.load(sess, [tag_constants.SERVING], self.Configs.export_dir)
                tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
                tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
                tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
                tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
                tensor_outputs = graph.get_tensor_by_name('Sigmoid:0')

                
                features = [self.convert_test_single_example(txt, max_seq_length, tokenizer) for txt in texts]
                input_ids = np.array([feature.input_ids for feature in features]).reshape(-1, max_seq_length)
                input_mask = np.array([feature.input_mask for feature in features]).reshape(-1, max_seq_length)
                label_ids = np.array([feature.label_id for feature in features])
                segment_ids = np.array([feature.segment_ids for feature in features]).reshape(-1, max_seq_length)

                pred = sess.run(tensor_outputs, feed_dict={
                        tensor_input_ids: input_ids,
                        tensor_input_mask: input_mask,
                        tensor_label_ids: label_ids,
                        tensor_segment_ids: segment_ids,
                    })
                
        return pred