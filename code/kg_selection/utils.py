# Copyright (c) Meta Platforms, Inc. and its affiliates.
import json
import math
import os
import random
import re
import time

from config import parameters as conf
from tqdm import tqdm

# Progress bar

TOTAL_BAR_LENGTH = 100.0
last_time = time.time()
begin_time = last_time
print(os.popen("stty size", "r").read())
_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)



def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f



def get_current_git_version():
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def write_log(log_file, s):
    print(s)
    with open(log_file, "a") as f:
        f.write(s + "\n")


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def read_txt(input_path, log_file):
    """Read a txt file into a list."""

    write_log(log_file, "Reading: %s" % input_path)
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.
    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).
    Returns:
      tokenized text.
    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    if conf.pretrained_model in ["bert", "finbert"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens


def read_examples(input_path, is_inference):
    """
    returns:
    {
        dialog_id:
        turn_id:
        context: util actions
        pos_snippets: [[ind, snippet1], [ind, snippet2], ...]
        neg_snippets: [[ind, snippet1], [ind, snippet2], ...]
        all_snippets: [[ind, snippet1], [ind, snippet2], ...] # for test time
    }
    """

    with open(input_path) as f:
        data_all = json.load(f)

    res = []
    for each_data in data_all:
        this_dialog_id = each_data["dialogue_id"]
        all_snippets = each_data["entity_passages_sents"]
        for ind, turn in enumerate(each_data["turns"]):
            if turn["speaker"] == "SYSTEM":
                this_turn_id = ind // 2

                if not is_inference:
                    if turn["enrich"]:
                        this_context = turn["tillaction"]
                        this_snippets = turn["kg_snippets"]

                        this_pos_snippets = []
                        this_neg_snippets = []
                        all_snippets = []

                        for each_query in each_data["entity_passages_sents"]:
                            for each_passage in each_data["entity_passages_sents"][
                                each_query
                            ]:
                                passage_title = each_passage[0]

                                for each_snippet in each_passage[1:]:
                                    if int(each_snippet[0]) in this_snippets:
                                        # pos snippets
                                        this_pos_snippets.append(
                                            [
                                                int(each_snippet[0]),
                                                passage_title + " " + each_snippet[1],
                                            ]
                                        )
                                    else:
                                        this_neg_snippets.append(
                                            [
                                                int(each_snippet[0]),
                                                passage_title + " " + each_snippet[1],
                                            ]
                                        )

                                    random.shuffle(this_neg_snippets)
                                    this_neg_snippets_select = this_neg_snippets[
                                        : len(this_pos_snippets) * conf.neg_rate
                                    ]

                        all_snippets = this_pos_snippets + this_neg_snippets
                        res.append(
                            {
                                "dialog_id": this_dialog_id,
                                "turn_id": this_turn_id,
                                "context": this_context,
                                "pos_snippets": this_pos_snippets,
                                "neg_snippets": this_neg_snippets_select,
                                "all_snippets": all_snippets,
                            }
                        )

                else:
                    # inference time for generated context
                    if conf.tillaction_gold:
                        this_context = turn["tillaction"]
                    else:
                        if "tillaction_pred" in turn:
                            this_context = turn["tillaction_pred"]

                    this_pos_snippets = []
                    this_neg_snippets = []
                    all_snippets = []

                    if conf.generate_all:
                        if conf.if_fill_train:
                            for each_query in each_data["entity_passages_sents"]:
                                for each_passage in each_data["entity_passages_sents"][
                                    each_query
                                ]:
                                    passage_title = each_passage[0]
                                    for each_snippet in each_passage[1:]:
                                        all_snippets.append(
                                            [
                                                int(each_snippet[0]),
                                                passage_title + " " + each_snippet[1],
                                            ]
                                        )
                            # negative turns
                            if not turn["enrich"]:
                                all_snippets = all_snippets[: conf.generate_all_neg_max]
                        else:
                            for each_query in turn["entity_passages_sents_pred"]:
                                for each_passage in turn["entity_passages_sents_pred"][
                                    each_query
                                ]:
                                    passage_title = each_passage[0]
                                    for each_snippet in each_passage[1:]:
                                        all_snippets.append(
                                            [
                                                int(each_snippet[0]),
                                                passage_title + " " + each_snippet[1],
                                            ]
                                        )

                        res.append(
                            {
                                "dialog_id": this_dialog_id,
                                "turn_id": this_turn_id,
                                "context": this_context,
                                "pos_snippets": this_pos_snippets,
                                "neg_snippets": this_neg_snippets,
                                "all_snippets": all_snippets,
                            }
                        )
                    else:
                        # if turn["enrich_pred"]:
                        for each_query in turn["entity_passages_sents_pred"]:
                            for each_passage in turn["entity_passages_sents_pred"][
                                each_query
                            ]:
                                passage_title = each_passage[0]
                                for each_snippet in each_passage[1:]:
                                    all_snippets.append(
                                        [
                                            int(each_snippet[0]),
                                            passage_title + " " + each_snippet[1],
                                        ]
                                    )
                        res.append(
                            {
                                "dialog_id": this_dialog_id,
                                "turn_id": this_turn_id,
                                "context": this_context,
                                "pos_snippets": this_pos_snippets,
                                "neg_snippets": this_neg_snippets,
                                "all_snippets": all_snippets,
                            }
                        )

    return data_all, res


def wrap_single_pair(
    tokenizer, question, context, label, max_seq_length, cls_token, sep_token
):
    """
    single pair of question, context, label feature
    """

    question_tokens = tokenize(tokenizer, question)
    this_gold_tokens = tokenize(tokenizer, context)

    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    tokens += this_gold_tokens
    segment_ids.extend([0] * len(this_gold_tokens))

    if len(tokens) > max_seq_length:
        tokens = tokens[: max_seq_length - 1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    this_input_feature = {
        "context": context,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label,
    }

    return this_input_feature


def convert_single_example(
    example, option, is_training, tokenizer, max_seq_length, cls_token, sep_token
):
    """convert all examples"""

    # {
    #     dialog_id:
    #     turn_id:
    #     context: util actions
    #     pos_snippets: [[ind, snippet1], [ind, snippet2], ...]
    #     neg_snippets: [[ind, snippet1], [ind, snippet2], ...]
    # }

    pos_features = []
    neg_features = []
    all_features = []

    context = example["context"]

    # print("###### question ######")
    # print(question)
    # print("\n")
    # print("Gold: ")
    # print("\n")

    # positive examples
    if is_training:
        for tmp in example["pos_snippets"]:

            each_pos_snippet = tmp[1]
            this_input_feature = wrap_single_pair(
                tokenizer,
                context,
                each_pos_snippet,
                1,
                max_seq_length,
                cls_token,
                sep_token,
            )

            this_input_feature["dialog_id"] = example["dialog_id"]
            this_input_feature["turn_id"] = example["turn_id"]
            this_input_feature["snippet_id"] = tmp[0]
            pos_features.append(this_input_feature)

        for tmp in example["neg_snippets"]:

            each_neg_snippet = tmp[1]
            this_input_feature = wrap_single_pair(
                tokenizer,
                context,
                each_neg_snippet,
                0,
                max_seq_length,
                cls_token,
                sep_token,
            )

            this_input_feature["dialog_id"] = example["dialog_id"]
            this_input_feature["turn_id"] = example["turn_id"]
            this_input_feature["snippet_id"] = tmp[0]
            neg_features.append(this_input_feature)

    else:
        for tmp in example["all_snippets"]:

            each_snippet = tmp[1]
            this_input_feature = wrap_single_pair(
                tokenizer,
                context,
                each_snippet,
                0,
                max_seq_length,
                cls_token,
                sep_token,
            )

            this_input_feature["dialog_id"] = example["dialog_id"]
            this_input_feature["turn_id"] = example["turn_id"]
            this_input_feature["snippet_id"] = tmp[0]
            all_features.append(this_input_feature)

    return pos_features, neg_features, all_features


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    option,
    is_training,
):
    """Converts a list of DropExamples into InputFeatures."""
    res = []
    res_neg = []
    res_all = []
    for (_, example) in tqdm(enumerate(examples)):
        features, features_neg, features_all = convert_single_example(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            option=option,
            is_training=is_training,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
        )

        res.extend(features)
        res_neg.extend(features_neg)
        res_all.extend(features_all)

    return res, res_neg, res_all


def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


class DataLoader:
    def __init__(self, is_training, data, batch_size=32, shuffle=True):
        """
        Main dataloader
        """
        self.data_pos = data[0]
        self.data_neg = data[1]
        self.data_all = data[2]
        self.batch_size = batch_size
        self.is_training = is_training

        if self.is_training:
            random.shuffle(self.data_neg)
            if conf.option == "tfidf":
                self.data = self.data_pos + self.data_neg
            else:
                num_neg = len(self.data_pos) * conf.neg_rate
                self.data = self.data_pos + self.data_neg[:num_neg]
        else:
            self.data = self.data_all

        self.data_size = len(self.data)
        self.num_batches = (
            int(self.data_size / batch_size)
            if self.data_size % batch_size == 0
            else int(self.data_size / batch_size) + 1
        )

        # print(self.num_batches)

        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # # drop last batch
        # if self.is_training:
        #     bound = self.num_batches - 1
        # else:
        #     bound = self.num_batches
        bound = self.num_batches
        if self.count < bound:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        if conf.option == "tfidf":
            random.shuffle(self.data)
        else:
            random.shuffle(self.data_neg)
            num_neg = len(self.data_pos) * conf.neg_rate
            self.data = self.data_pos + self.data_neg[:num_neg]
            random.shuffle(self.data)
        return

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)

        batch_data = {
            "input_ids": [],
            "input_mask": [],
            "segment_ids": [],
            "dialog_id": [],
            "turn_id": [],
            "snippet_id": [],
            "label": [],
        }
        for each_data in self.data[start_index:end_index]:

            batch_data["input_ids"].append(each_data["input_ids"])
            batch_data["input_mask"].append(each_data["input_mask"])
            batch_data["segment_ids"].append(each_data["segment_ids"])
            batch_data["dialog_id"].append(each_data["dialog_id"])
            batch_data["turn_id"].append(each_data["turn_id"])
            batch_data["snippet_id"].append(each_data["snippet_id"])
            batch_data["label"].append(each_data["label"])

        return batch_data


def cleanhtml(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext


def retrieve_evaluate(
    all_logits,
    all_dialog_id,
    all_turn_id,
    all_snippet_id,
    output_prediction_file,
    ori_file,
    topn,
    is_inference,
):
    """
    save results to file. calculate recall
    """

    res_dialog = {}

    for this_logit, this_dialog_id, this_turn_id, this_snippet_id in zip(
        all_logits, all_dialog_id, all_turn_id, all_snippet_id
    ):

        this_score = _compute_softmax(this_logit)
        this_ind = str(this_dialog_id) + "_" + str(this_turn_id)
        if this_ind not in res_dialog:
            res_dialog[this_ind] = []
        if this_snippet_id not in res_dialog[this_ind]:
            res_dialog[this_ind].append(
                {
                    "score": this_score[1],
                    "snippet": this_snippet_id,
                }
            )

    with open(ori_file) as f:
        data_all = json.load(f)

    # take top ten
    all_recall_3 = 0.0
    all_kg_chitchat = 0

    for data in data_all:
        for ind, turn in enumerate(data["turns"]):
            if turn["speaker"] == "SYSTEM":
                if not is_inference:
                    if turn["enrich"]:
                        this_ind = data["dialogue_id"] + "_" + str(ind // 2)

                        this_res = res_dialog[this_ind]

                        sorted_dict = sorted(
                            this_res, key=lambda kv: kv["score"], reverse=True
                        )

                        sorted_dict = sorted_dict[:topn]

                        all_kg_chitchat += 1
                        gold_inds = turn["kg_snippets"]

                        correct_3 = 0

                        retrieved_snippets = []

                        for tmp in sorted_dict[:3]:
                            retrieved_snippets.append([tmp["snippet"], tmp["score"]])
                            if tmp["snippet"] in gold_inds:
                                correct_3 += 1

                        all_recall_3 += float(correct_3) / len(gold_inds)

                        turn["retrieved"] = retrieved_snippets

                else:
                    # inference model
                    if not conf.generate_all:
                        if turn["enrich_pred"]:
                            this_ind = data["dialogue_id"] + "_" + str(ind // 2)

                            # some turns have no retrieved results
                            retrieved_snippets = []
                            if this_ind in res_dialog:
                                this_res = res_dialog[this_ind]

                                sorted_dict = sorted(
                                    this_res, key=lambda kv: kv["score"], reverse=True
                                )

                                for tmp in sorted_dict[:topn]:
                                    # if tmp["score"] >= conf.thresh:
                                    retrieved_snippets.append(tmp["snippet"])

                            else:
                                print(this_ind)

                            if len(retrieved_snippets) == 0:
                                retrieved_snippets.append(0)
                            turn["retrieved"] = retrieved_snippets

                    else:
                        this_ind = data["dialogue_id"] + "_" + str(ind // 2)

                        # some turns have no retrieved results
                        retrieved_snippets = []
                        if this_ind in res_dialog:
                            this_res = res_dialog[this_ind]

                            sorted_dict = sorted(
                                this_res, key=lambda kv: kv["score"], reverse=True
                            )

                            for tmp in sorted_dict[:topn]:
                                # if tmp["score"] >= conf.thresh:
                                retrieved_snippets.append(tmp["snippet"])

                        else:
                            print(this_ind)

                        if len(retrieved_snippets) == 0:
                            retrieved_snippets.append(0)
                        turn["retrieved"] = retrieved_snippets

    with open(output_prediction_file, "w") as f:
        json.dump(data_all, f, indent=4)

    if not is_inference:
        res_3 = all_recall_3 / all_kg_chitchat

        res = "Top 3: " + str(res_3) + "\n"
    else:
        res = "finished"

    return res


if __name__ == "__main__":

    root_path = "/mnt/george_bhd/zhiyuchen/"
    outputs = root_path + "outputs/"

    json_in = (
        outputs + "test_20210408011241/results/loads/1/valid/nbest_predictions.json"
    )
    retrieve_evaluate(json_in)
