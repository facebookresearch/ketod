# Copyright (c) Meta Platforms, Inc. and its affiliates.
import json
from tqdm import tqdm


from utils import get_kg_snippets_dict
'''
merge kg selection results with original
'''


def merge_train(json_in, json_out, topn_snippets=3, is_test=True):
    '''
    merge train or dev
    '''

    with open(json_in) as f:
        data_all = json.load(f)


    for each_data in tqdm(data_all):

        all_kg_snippets_dict = get_kg_snippets_dict(each_data)

        for turn in each_data["turns"]:
            if turn["speaker"] == "SYSTEM":

                this_retrieved_kg_text = []
                if turn["enrich"]:
                    if not is_test:
                        this_retrieved_kg_text = turn["kg_snippets_text"][:]

                        retrieved_ind = 0
                        while len(this_retrieved_kg_text) < topn_snippets and retrieved_ind < len(turn["retrieved"]):
                            # make up with the retrieved ones
                            this_added_ind = turn["retrieved"][retrieved_ind]
                            if this_added_ind not in turn["kg_snippets"]:
                                added_kg_text = all_kg_snippets_dict[turn["retrieved"][retrieved_ind]]
                                this_retrieved_kg_text.append(added_kg_text)
                            retrieved_ind += 1
                    else:
                        for each_added in turn["retrieved"]:
                            this_retrieved_kg_text.append(all_kg_snippets_dict[each_added])

                else:
                    for each_added in turn["retrieved"]:
                        this_retrieved_kg_text.append(all_kg_snippets_dict[each_added])

                turn["merge_retrieved"] = this_retrieved_kg_text


    with open(json_out, "w") as f:
        json.dump(data_all, f, indent=4)





root = "/data/users/zhiyuchen/outputs/"
tgt = "/data/users/zhiyuchen/todkg_dataset/runs/"

# # train
# json_in = root + "inference_only_20210818034712_kg_select_bert_base_train/results/test/predictions.json"
# json_out = tgt + "model1/" + "train_final.json"

# merge_train(json_in, json_out, topn_snippets=3, is_test=False)

# # dev
# json_in = root + "inference_only_20210818034804_kg_select_bert_base_dev/results/test/predictions.json"
# json_out = tgt + "model1/" + "dev_final.json"

# merge_train(json_in, json_out, topn_snippets=3, is_test=False)


# test
json_in = root + "inference_only_20210818034853_kg_select_bert_base_test/results/test/predictions.json"
json_out = tgt + "model1/" + "test_retrieved.json"

merge_train(json_in, json_out, topn_snippets=3, is_test=True)
