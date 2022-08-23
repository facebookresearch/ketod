'''

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
'''
import argparse
import collections
import json
import os
import re
import sys

# generate ketod dataset
# python 3.9

def gen_ketod(json_in, sgd_folder_in, json_out, mode="train"):
    '''
    generate the ketod dataset
    combining our annotation and the sgd data
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    all_sgd_train = {}
    sgd_folder = os.path.join(sgd_folder_in, "train")
    for filename in os.listdir(sgd_folder):
        if "dialogues" in filename:
            with open(os.path.join(sgd_folder, filename)) as f_in:
                this_data = json.load(f_in)
                for each_data in this_data:
                    assert each_data["dialogue_id"] not in all_sgd_train

                    all_sgd_train[each_data["dialogue_id"]] = each_data

    all_sgd = {}
    sgd_folder = os.path.join(sgd_folder_in, mode)
    for filename in os.listdir(sgd_folder):
        if "dialogues" in filename:
            with open(os.path.join(sgd_folder, filename)) as f_in:
                this_data = json.load(f_in)
                for each_data in this_data:
                    assert each_data["dialogue_id"] not in all_sgd

                    all_sgd[each_data["dialogue_id"]] = each_data

    for each_data in data:
        if each_data["dialogue_id"] in all_sgd:
            this_sgd = all_sgd[each_data["dialogue_id"]]
        else:
            this_sgd = all_sgd_train[each_data["dialogue_id"]]
        this_final_turns = []
        for sgd_turn, ketod_turn in zip(this_sgd["turns"], each_data["turns"]):
            final_turn = sgd_turn | ketod_turn
            this_final_turns.append(final_turn)

        each_data["turns"] = this_final_turns

    print(len(data))

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)




if __name__ == '__main__':


    root = "path_to_your_project"

    ketod_release = root + "ketod_release/"

    ketod_release_train = ketod_release + "train_ketod.json"
    ketod_release_dev = ketod_release + "dev_ketod.json"
    ketod_release_test = ketod_release + "test_ketod.json"

    sgd = root + "dstc8-schema-guided-dialogue/"

    train_final = ketod_release + "train.json"
    dev_final = ketod_release + "dev.json"
    test_final = ketod_release + "test.json"

    # generate ketod dataset
    gen_ketod(ketod_release_train, sgd, train_final, mode="train")
    gen_ketod(ketod_release_dev, sgd, dev_final, mode="dev")
    gen_ketod(ketod_release_test, sgd, test_final, mode="test")
