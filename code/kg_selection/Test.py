#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and its affiliates.
"""
Main script
"""
from tqdm import tqdm
import os
import json
from datetime import datetime
import time
from utils import read_examples, convert_examples_to_features, write_log, DataLoader, retrieve_evaluate
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim


from Model import Bert_model

if conf.pretrained_model == "bert":
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)


saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
# model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + conf.model_save_name
model_dir = os.path.join(
    conf.output_path, 'inference_only_' + model_dir_name)
results_path = os.path.join(model_dir, "results")
os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')
test_feature_file = os.path.join(results_path, 'this_features.txt')


test_data, test_examples = \
    read_examples(input_path=conf.test_file, is_inference=True)

kwargs = {"examples": test_examples,
          "tokenizer": tokenizer,
          "option": conf.option,
          "is_training": False,
          "max_seq_length": conf.max_seq_length,
          }

kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)

with open(test_feature_file, "w") as f:
    json.dump(list(test_features), f, indent=4)

# with open(test_feature_file) as f:
#     test_features = json.load(f)


def generate(data_ori, data, model, ksave_dir, mode='valid'):

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, shuffle=False)

    k = 0
    all_logits = []
    all_dialog_id = []
    all_turn_id = []
    all_snippet_id = []

    with torch.no_grad():
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            label = x['label']
            dialog_id = x["dialog_id"]
            turn_id = x["turn_id"]
            snippet_id = x["snippet_id"]

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)


            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            logits = model(True, input_ids, input_mask,
                           segment_ids, device=conf.device)

            all_logits.extend(logits.tolist())
            all_dialog_id.extend(dialog_id)
            all_turn_id.extend(turn_id)
            all_snippet_id.extend(snippet_id)

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")

    if mode == "valid":
        print_res = retrieve_evaluate(
            all_logits, all_dialog_id, all_turn_id, all_snippet_id, output_prediction_file, conf.valid_file, topn=conf.topn, is_inference=True)
    else:
        print_res = retrieve_evaluate(
            all_logits, all_dialog_id, all_turn_id, all_snippet_id, output_prediction_file, conf.test_file, topn=conf.topn, is_inference=True)

    write_log(log_file, print_res)
    print(print_res)
    return


def generate_test():
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,)

    model = nn.DataParallel(model)
    model.to(conf.device)
    model.load_state_dict(torch.load(conf.saved_model_path))
    model.eval()
    generate(test_data, test_features, model, results_path, mode='test')




if __name__ == '__main__':

    generate_test()
