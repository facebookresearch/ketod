# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Test simpleToD LM

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import argparse
import numpy as np
from torch import nn
import os
import json
from utils import write_log
from tqdm import tqdm

'''
params list
'''

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action="store_true", help="avoid using CUDA when available")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--saved_model_path", type=str, default="output", help="path to save model")
parser.add_argument("--output_path", type=str, help="output path")
parser.add_argument("--model_dir_name", type=str, help="model inference directory name")


parser.add_argument("--gold_action", action="store_true", help="whether gold action")
parser.add_argument("--gold_kg", action="store_true", help="whether gold kg")
parser.add_argument("--gold_decision", action="store_true", help="whether gold decision")
parser.add_argument("--test_input_gold_action", type=str, help="input test text file gold action, each line corresponding to one instance")
parser.add_argument("--test_input_gold_kg", type=str, help="input test text file gold kg, each line corresponding to one instance")
parser.add_argument("--test_input_gold_decision", type=str, help="input test text file gold kg, each line corresponding to one instance")


parser.add_argument("--test_input", type=str, help="input test text file, each line corresponding to one instance")
parser.add_argument("--test_oracle_input", type=str, help="input test text file oracle, each line corresponding to one instance")
parser.add_argument("--test_input_original", type=str, help="input test file, original json")

parser.add_argument("--test_inter", type=str, help="intermediate file for kg selection")
parser.add_argument("--test_inter_res", type=str, help="intermediate result file for kg selection")

parser.add_argument("--en_schema", type=str, help="schema file")

parser.add_argument("--num_passages", type=int, default=2, help="num of passages for each entity")
parser.add_argument("--num_para", type=int, default=2, help="num of paragraphs for each passage")


parser.add_argument("--eos_token_id", type=int, default=None, help="eos token id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")

parser.add_argument("--max_seq_len", type=int, default=512, help="max sequence length")



def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

saved_model_path = os.path.join(args.output_path, args.saved_model_path)

model_dir = os.path.join(
    args.output_path, 'inference_only_' + args.model_dir_name)
results_path = os.path.join(model_dir, "results")
if not os.path.isdir(results_path):
    os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')

result_file_belief = os.path.join(results_path, 'result_belief.json')
result_file_action = os.path.join(results_path, 'result_action.json')
result_file_knowledge = os.path.join(results_path, 'result_knowledge.json')
result_file_final = os.path.join(results_path, 'result_final.json')


set_seed(args)

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', \
                                                            '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', \
                                                            '<|task|>', '<|endoftask|>', '<|chitchat|>', '<|nochitchat|>', '<|endofdecision|>', '<|knowledge|>', '<|endofknowledge|>', '<|dbresults|>', '<|endofdbresults|>']})

model.resize_token_embeddings(len(tokenizer))
model = nn.DataParallel(model)
model.to(args.device)
model.load_state_dict(torch.load(saved_model_path))
print("model loaded")
model.eval()


with open(args.test_input_original) as f:
    data_ori = json.load(f)

with open(args.test_input, "r") as f:
    prompts = f.read().strip().split("\n")
    print("Num test: ", len(prompts))

with open(args.test_oracle_input, "r") as f:
    prompts_oracle = f.read().strip().split("\n")
    print("Num test oracle: ", len(prompts_oracle))

batch_size = args.batch_size
num_batch = len(prompts) // batch_size
if batch_size * num_batch < len(prompts):
    num_batch += 1




# keep track of all input parameters
write_log(log_file, "####################INPUT PARAMETERS###################")
for attr in args.__dict__:
    value = args.__dict__[attr]
    write_log(log_file, attr + " = " + str(value))
write_log(log_file, "#######################################################")



def decode_seq(prompts, eos_token_id):
    # eos_token_id as list
    ret = []
    with torch.no_grad():
        for batch in tqdm(range(num_batch)):

            prompt_text = prompts[batch * batch_size: (batch + 1) * batch_size]

            encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

            input_ids = torch.tensor(encodings_dict['input_ids'])
            attn_mask = torch.tensor(encodings_dict['attention_mask'])

            seq_len = len(input_ids[0])

            num_tokens_to_produce = 1024 - seq_len
            pad_token_id = tokenizer.pad_token_id
            # eos_token_id = args.eos_token_id
            if eos_token_id is None:
                eos_token_id = tokenizer.eos_token_id
            eos_not_in_sents = torch.ones(input_ids.shape[0]).long()

            last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
            start_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size + len(tokenizer.additional_special_tokens)).unsqueeze(1)

            position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
            for i, position_ids_slice in enumerate(position_ids):
                position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

            input_ids = input_ids.to(args.device)
            attn_mask = attn_mask.to(args.device)
            eos_not_in_sents = eos_not_in_sents.to(args.device)
            start_idx = start_idx.to(args.device)
            position_ids = position_ids.to(args.device)


            # past = None
            # generated = torch.tensor(encodings_dict['input_ids']).to(args.device)

            # for step in range(num_tokens_to_produce):
            #     outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids, past_key_values=past, return_dict=True)

            #     if step == 0:
            #         next_token_logits = outputs.logits.gather(1, start_idx).squeeze(1)
            #     else:
            #         next_token_logits = outputs.logits[:, -1, :]

            #     next_tokens = torch.argmax(next_token_logits, dim=-1)


            #     eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())

            #     tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)

            #     generated = torch.cat([generated, tokens_to_add.unsqueeze(-1)], dim=-1)

            #     # input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            #     # attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long().to(args.device)], dim=1)
            #     # position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

            #     input_ids = tokens_to_add.unsqueeze(-1)
            #     attn_mask = torch.ones((attn_mask.shape[0], 1)).long().to(args.device)
            #     position_ids = (position_ids[:, -1] + 1).unsqueeze(-1)

            #     past = outputs.past_key_values

            #     if torch.max(eos_not_in_sents) == 0:
            #         break

            # ret += [tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=True).replace("<|endoftext|>", "") for output in generated]


            for step in range(num_tokens_to_produce):
                outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids)

                if step == 0:
                    next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
                else:
                    next_token_logits = outputs[0][:, -1, :]

                next_tokens = torch.argmax(next_token_logits, dim=-1)

                eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())

                tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)

                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long().to(args.device)], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

                if torch.max(eos_not_in_sents) == 0:
                    break


            ret += [tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=True).replace("<|endoftext|>", "") for output in input_ids]

    return ret


# step 1: decode belief state
def gen_belief():
    state_eos = tokenizer.encode('<|endofbelief|>')[0]

    # # reorder prompts to make inference faster
    # reorder_prompts = {}
    # for ind, each_tmp in enumerate(prompts):
    #     reorder_prompts[ind] = each_tmp

    # reorder_prompts = sorted(reorder_prompts.items(), key=lambda item: len(item[1].split(" ")))
    # for tmp in reorder_prompts:
    #     print(tmp)
    # reorder_ind = [tmp[0] for tmp in reorder_prompts]
    # reorder_input = [tmp[1] for tmp in reorder_prompts]

    # decode_belief_mess = decode_seq(reorder_input, eos_token_id = state_eos)

    # # reorder back to test order
    # decode_belief = []
    # for ind in range(len(decode_belief_mess)):
    #     decode_belief.append(decode_belief_mess[reorder_ind.index(ind)])


    decode_belief = decode_seq(prompts, eos_token_id = state_eos)
    state_res = []
    for row in decode_belief:
        row = row.replace(" <PAD>", "")
        row = row.strip() + ' <|endofbelief|>'
        state_res.append(row)

    with open(result_file_belief, "w") as f:
        json.dump(state_res, f, indent=4)

    return state_res


# step 2: add oracle db result, gen action
def gen_action():

    with open(result_file_belief, "r") as f:
        prompts_belief = json.load(f)

    input_belief = []
    for each_belief, each_oracle in zip(prompts_belief, prompts_oracle):
        this_oracle_db = each_oracle.split('<|dbresults|>')[1].split('<|endofdbresults|>')[0].strip()
        each_belief += (" <|dbresults|> " + this_oracle_db + ' <|endofdbresults|>')
        input_belief.append(each_belief)

    action_eos = tokenizer.encode('<|endofaction|>')[0]
    decode_action = decode_seq(input_belief, eos_token_id = action_eos)
    action_res = []
    for row in decode_action:
        row = row.replace(" <PAD>", "")
        row = row.strip() + ' <|endofaction|>'
        action_res.append(row)

    with open(result_file_action, "w") as f:
        json.dump(action_res, f, indent=4)

    return action_res


# step 3: combine with original test file, as the input to kg selector
# TODO: add knowledge retrieval
def gen_kg_selection_input():

    from drqa import retriever
    from drqa.retriever import DEFAULTS
    from drqa.retriever.doc_db import DocDB
    ranker = retriever.get_class('tfidf')(DEFAULTS["tfidf_path"])
    this_db = DocDB(db_path=DEFAULTS["db_path"])

    import nltk.data
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


    if args.gold_action:
        with open(args.test_input_gold_action) as f:
            prompts_action = f.read().strip().split("\n")
            print(len(prompts_action))

    else:
        with open(result_file_action) as f:
            prompts_action = json.load(f)


    with open(args.en_schema) as f:
        en_schema = json.load(f)

    # for checking if is entity
    schema_list = set()
    slot_list = set()
    for service in en_schema:
        # service = service.split("_")[0].lower()
        schema_list.add(service.split("_")[0].lower())

        for slot in en_schema[service]:
            slot_list.add(slot["name"])



    ind = 0
    for each_data in data_ori:
        for turn in each_data["turns"]:
            if turn["speaker"] == "SYSTEM":
                # turn["tillaction_generated"] = prompts_action[ind]
                # ind += 1


                turn["tillaction_pred"] = prompts_action[ind]

                # print("\n")
                # print("###############")
                # print(turn["tillaction_pred"])

                query_list = []

                # entities in belief
                if '<|belief|>' in prompts_action[ind] and '<|endofbelief|>' in prompts_action[ind]:
                    beliefs_text = prompts_action[ind].split('<|belief|>')[1].split('<|endofbelief|>')[0].strip()
                    if beliefs_text:
                        for each_belief in beliefs_text.split(","):
                            each_belief = each_belief.strip().split(" ")
                            if len(each_belief) >= 3:
                                service = each_belief[0].strip()
                                slot_name = each_belief[1].strip()
                                entity_name = " ".join(each_belief[2:]).strip()

                                # print("Each belief")
                                # print(each_belief)
                                # print(service)
                                # print(slot_name)
                                # print(entity_name)
                                # print("each belief end")

                                if service in schema_list and slot_name in slot_list:

                                    slot_name = slot_name.replace("_", " ")
                                    query = " : ".join([service, slot_name, entity_name])
                                    if query not in query_list:
                                        query_list.append(query)

                # entities in action
                if '<|action|>' in prompts_action[ind] and '<|endofaction|>' in prompts_action[ind]:
                    actions_text = prompts_action[ind].split('<|action|>')[1].split('<|endofaction|>')[0].strip()
                    if actions_text:
                        for each_action in actions_text.split(","):
                            each_action = each_action.strip().split(" ")
                            if len(each_action) >= 4:
                                service = each_action[0].strip()
                                slot_name = each_action[2].strip()
                                entity_name = " ".join(each_action[3:]).strip()

                                if service in schema_list and slot_name in slot_list:

                                    slot_name = slot_name.replace("_", " ")
                                    query = " : ".join([service, slot_name, entity_name])
                                    if query not in query_list:
                                        query_list.append(query)

                sents_ind = 1
                this_res_dict_sents = {}
                for query in query_list:
                    # retrieve passages
                    this_res_dict_sents[query] = []
                    # print(query)
                    doc_names, doc_scores = ranker.closest_docs(query.replace(" :", ""), args.num_passages)

                    for each_doc in doc_names:
                        this_text = this_db.get_doc_text(each_doc)
                        res_text_list = []
                        for each_para in this_text.split("\n"):
                            if each_para.strip() != "":
                                res_text_list.append(each_para.strip())

                        res_text_list = res_text_list[:args.num_para + 1] # plus article title

                        res_sents_text_list = [res_text_list[0]]
                        this_sents_list = sent_tokenizer.tokenize(" ".join(res_text_list[1:]))
                        for each_sent in this_sents_list:
                            res_sents_text_list.append([sents_ind, each_sent])
                            sents_ind += 1

                        this_res_dict_sents[query].append(res_sents_text_list)

                turn["entity_query_pred"] = query_list
                turn["entity_passages_sents_pred"] = this_res_dict_sents

                ind += 1


    with open(args.test_inter, "w") as f:
        json.dump(data_ori, f, indent=4)



# step 4: add selected knowledge
def add_selected_kg():


    with open(result_file_action, "r") as f:
        prompts_action = json.load(f)

    with open(args.test_inter_res, "r") as f:
        data_all = json.load(f)


    res = []
    ind = 0
    for each_data in data_all:
        for turn in each_data["turns"]:
            if turn["speaker"] == "SYSTEM":
                this_pre_gen = prompts_action[ind]
                ind += 1

                selected_kg_snippets = turn["retrieved"]
                this_kg_text = []

                for each_query in turn["entity_passages_sents_pred"]:
                    for each_passage in turn["entity_passages_sents_pred"][each_query]:
                        passage_title = each_passage[0]

                        for each_snippet in each_passage[1:]:
                            if int(each_snippet[0]) in selected_kg_snippets:
                                this_kg_text.append(passage_title + " " + each_snippet[1])

                res.append(this_pre_gen + ' <|knowledge|> ' + " ".join(this_kg_text) + ' <|endofknowledge|>')


    assert ind == len(prompts_action)

    with open(result_file_knowledge, "w") as f:
        json.dump(res, f, indent=4)



# step 5: gen response
def gen_response():

    if args.gold_kg:
        with open(args.test_input_gold_kg) as f:
            prompts_action = f.read().strip().split("\n")
            print(len(prompts_action))

    elif args.gold_decision:
        with open(args.test_input_gold_decision) as f:
            prompts_action = f.read().strip().split("\n")
            print(len(prompts_action))

    else:
        with open(result_file_knowledge) as f:
            prompts_action = json.load(f)


    # with open(result_file_knowledge, "r") as f:
    #     prompts_action = json.load(f)

    response_eos = tokenizer.encode('<|endofresponse|>')[0]
    decode_response = decode_seq(prompts_action, eos_token_id = response_eos)
    response_res = []
    for row in decode_response:
        row = row.replace(" <PAD>", "")
        row = row.strip() + ' <|endofresponse|>'
        response_res.append(row)

    with open(result_file_final, "w") as f:
        json.dump(response_res, f, indent=4)

    return response_res


# step 5: gen response
def gen_response_retrieved_kg_gold_decision():

    with open(args.test_oracle_input) as f:
        prompts_gold = f.read().strip().split("\n")
    with open(args.test_input_gold_kg) as f:
        prompts_kg = f.read().strip().split("\n")

    inputs = []
    for gold, kg in zip(prompts_gold, prompts_kg):
        this_decision = gold.split('<|endofknowledge|>')[1].strip().split('<|response|>')[0].strip()
        inputs.append(" ".join([kg, this_decision]))


    # with open(result_file_knowledge, "r") as f:
    #     prompts_action = json.load(f)

    response_eos = tokenizer.encode('<|endofresponse|>')[0]
    decode_response = decode_seq(inputs, eos_token_id = response_eos)
    response_res = []
    for row in decode_response:
        row = row.replace(" <PAD>", "")
        row = row.strip() + ' <|endofresponse|>'
        response_res.append(row)

    with open(result_file_final, "w") as f:
        json.dump(response_res, f, indent=4)

    return response_res




# ### full test pipeline
# # step 1:
# gen_belief()

# # step 2:
# gen_action()

# # step 3:
# gen_kg_selection_input()

# # step 4:
# add_selected_kg()

# # step 5:
# gen_response()



### test with gold kg
gen_response()


# ### generated kg, gold decision
# gen_response_retrieved_kg_gold_decision()
