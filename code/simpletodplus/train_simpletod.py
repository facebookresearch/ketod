# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Training simpleToD LM

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn
import argparse
import numpy as np
import os
import random
import time
from datetime import datetime
from utils import write_log
import torch.optim as optim
from tqdm import tqdm

'''
params list
'''

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action="store_true", help="avoid using CUDA when available")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--model_save_name", type=str, default="output", help="path to save model")
parser.add_argument("--output_path", type=str, help="output path")

parser.add_argument("--input", type=str, help="input text file, each line corresponding to one instance")
parser.add_argument("--dev_input", type=str, help="input dev text file, each line corresponding to one instance")

parser.add_argument("--eos_token_id", type=int, default=None, help="eos token id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--max_epoch", type=int, default=50, help="epoch")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning_rate")
parser.add_argument("--report_loss", type=int, default=500, help="steps to report loss")
parser.add_argument("--report", type=int, default=1000, help="steps to save model")

parser.add_argument("--max_seq_len", type=int, default=512, help="max sequence length")

parser.add_argument("--neg_sample", action="store_true", help="whether to do negative sampling")
parser.add_argument("--neg_sample_rate", type=int, default=3, help="rate of nochitchat to chitchat")




def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

model_dir_name = args.model_save_name + "_" + \
    datetime.now().strftime("%Y%m%d%H%M%S")
model_dir = os.path.join(args.output_path, model_dir_name)
results_path = os.path.join(model_dir, "results")
saved_model_path = os.path.join(model_dir, "saved_model")
os.makedirs(saved_model_path, exist_ok=False)
os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')

set_seed(args)

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')

tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', \
                                                            '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', \
                                                            '<|task|>', '<|endoftask|>', '<|chitchat|>', '<|nochitchat|>', '<|endofdecision|>', '<|knowledge|>', '<|endofknowledge|>', '<|dbresults|>', '<|endofdbresults|>']})

model.resize_token_embeddings(len(tokenizer))
model = nn.DataParallel(model)
model.to(args.device)
optimizer = optim.Adam(model.parameters(), args.learning_rate)
model.train()


def lm_pass(model, prompt_text, dev=True):
    '''
    language modeling function
    '''

    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'])
    attn_mask = torch.tensor(encodings_dict['attention_mask'])

    seq_len = len(input_ids[0])

    # if dev:
    #     print(seq_len)

    if seq_len > args.max_seq_len:
        # truncate to max seq len
        input_ids = torch.split(input_ids, args.max_seq_len, dim=1)[0]
        attn_mask = torch.split(attn_mask, args.max_seq_len, dim=1)[0]
        seq_len = len(input_ids[0])


    last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1

    position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

    input_ids = input_ids.to(args.device)
    attn_mask = attn_mask.to(args.device)
    labels = input_ids.to(args.device)
    position_ids = position_ids.to(args.device)


    outputs = model(input_ids=input_ids, labels=labels, position_ids=position_ids, return_dict=True)
    loss = outputs.loss.sum()

    return loss



with open(args.input, "r") as f:
    prompts_ori = f.read().strip().split("\n")
    print("Num train: ", len(prompts_ori))

with open(args.dev_input, "r") as f:
    dev_prompts = f.read().strip().split("\n")
    print("Num dev: ", len(dev_prompts))

prompts_chitchat = []
prompts_nochitchat = []

for tmp in prompts_ori:
    if "<|chitchat|>" in tmp:
        prompts_chitchat.append(tmp)
    else:
        prompts_nochitchat.append(tmp)

print("Chitchat: ", len(prompts_chitchat))
print("Nochitchat: ", len(prompts_nochitchat))
num_train = len(prompts_chitchat) * (args.neg_sample_rate + 1)


batch_size = args.batch_size
num_batch = num_train // batch_size if num_train % batch_size == 0 \
else num_train // batch_size + 1

num_batch_dev = len(dev_prompts) // batch_size if len(dev_prompts) % batch_size == 0 \
else len(dev_prompts) // batch_size + 1

start_time = time.time()
k = 0
record_k = 0
record_loss = 0.0



# keep track of all input parameters
write_log(log_file, "####################INPUT PARAMETERS###################")
for attr in args.__dict__:
    value = args.__dict__[attr]
    write_log(log_file, attr + " = " + str(value))
write_log(log_file, "#######################################################")

for _ in range(args.max_epoch):

    # subsample negative examples?
    # random.shuffle(prompts_chitchat)
    random.shuffle(prompts_nochitchat)
    if args.neg_sample:
        prompts = prompts_chitchat + prompts_nochitchat[:len(prompts_chitchat)*args.neg_sample_rate]
    else:
        prompts = prompts_chitchat + prompts_nochitchat
    random.shuffle(prompts)

    for batch in range(num_batch):

        prompt_text = prompts[batch * batch_size: (batch + 1) * batch_size]

        model.zero_grad()
        optimizer.zero_grad()

        loss = lm_pass(model, prompt_text, dev=False)

        loss.backward()
        optimizer.step()

        record_loss += loss.item()
        record_k += 1
        k += 1

        if k > 1 and k % args.report_loss == 0:
            write_log(log_file, "%d : loss = %.3f" %
                        (k, record_loss / record_k))
            record_loss = 0.0
            record_k = 0

        if k > 1 and k % args.report == 0:
            print("Round: ", k / args.report)
            model.eval()
            cost_time = time.time() - start_time
            write_log(log_file, "%d : time = %.3f " %
                        (k // args.report, cost_time))
            start_time = time.time()
            if k // args.report >= 1:
                print("Val test")
                # save model
                saved_model_path_cnt = os.path.join(saved_model_path, 'loads', str(k // args.report))
                os.makedirs(saved_model_path_cnt, exist_ok=True)
                torch.save(model.state_dict(), saved_model_path_cnt + "/model.pt")

                eval_loss = 0.0
                with torch.no_grad():
                    for dev_batch in tqdm(range(num_batch_dev)):
                        dev_prompt_text = dev_prompts[dev_batch * batch_size: (dev_batch + 1) * batch_size]
                        eval_loss += lm_pass(model, dev_prompt_text).item()

                write_log(log_file, "This round eval loss = %.3f " % eval_loss)

            model.train()
