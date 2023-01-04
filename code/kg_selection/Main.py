#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and its affiliates.
"""
Main script
"""
from tqdm import tqdm
import os
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

# create output paths
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(
        conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')




train_data, train_examples = \
    read_examples(input_path=conf.train_file, is_inference=False)

valid_data, valid_examples = \
    read_examples(input_path=conf.valid_file, is_inference=False)

test_data, test_examples = \
    read_examples(input_path=conf.test_file, is_inference=False)

kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "option": conf.option,
          "is_training": True,
          "max_seq_length": conf.max_seq_length,
          }


train_features = convert_examples_to_features(**kwargs)

kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,)

    model = nn.DataParallel(model)
    model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()

    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, shuffle=True)

    k = 0
    record_k = 0
    start_time = time.time()
    record_loss = 0.0

    for _ in range(conf.epoch):
        train_iterator.reset()
        for x in train_iterator:

            # results_path_cnt = os.path.join(
            #     results_path, 'loads', str(k // conf.report))
            # os.makedirs(results_path_cnt, exist_ok=True)
            # validation_result = evaluate(
            #     train_examples, train_features, model, results_path_cnt, 'valid')

            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            label = torch.tensor(x['label']).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            this_logits = model(True, input_ids, input_mask,
                                segment_ids, device=conf.device)

            this_loss = criterion(
                this_logits.view(-1, this_logits.shape[-1]), label.view(-1))

            this_loss = this_loss.sum()
            record_loss += this_loss.item() * 100
            record_k += 1
            k += 1

            this_loss.backward()
            optimizer.step()

            if k > 1 and k % conf.report_loss == 0:
                write_log(log_file, "%d : loss = %.3f" %
                          (k, record_loss / record_k))
                record_loss = 0.0
                record_k = 0

            if k > 1 and k % conf.report == 0:
                print("Round: ", k / conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, "%d : time = %.3f " %
                          (k // conf.report, cost_time))
                start_time = time.time()
                if k // conf.report >= 1:
                    print("Val test")
                    # save model
                    saved_model_path_cnt = os.path.join(saved_model_path, 'loads', str(k // conf.report))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    torch.save(model.state_dict(), saved_model_path_cnt + "/model.pt")

                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_path_cnt, exist_ok=True)
                    validation_result = evaluate(
                        valid_examples, valid_features, model, results_path_cnt, 'valid')
                    # write_log(log_file, validation_result)

                model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):

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
            all_logits, all_dialog_id, all_turn_id, all_snippet_id, output_prediction_file, conf.valid_file, topn=conf.topn, is_inference=False)
    else:
        print_res = retrieve_evaluate(
            all_logits, all_dialog_id, all_turn_id, all_snippet_id, output_prediction_file, conf.test_file, topn=conf.topn, is_inference=False)

    write_log(log_file, print_res)
    print(print_res)
    return


if __name__ == '__main__':
    train()
