# Copyright (c) Meta Platforms, Inc. and its affiliates.

# generate non-delex data

import json
import copy
import random
import argparse

root = "/data/users/zhiyuchen/todkg_dataset/processed/"
tgt = "/data/users/zhiyuchen/todkg_dataset/runs/kg_select/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train_final.json", type=str, required=False, help="data file name")
    args = parser.parse_args()


    with open(root + args.data, "r", encoding='utf8') as f:
        data = json.load(f)
    i = 0
    while i < len(data):

        for j in range(1, len(data[i]["turns"]), 2):
            domain = data[i]["turns"][j]["frames"][0]["service"].split("_")[0].lower()
            assert(data[i]["turns"][j]["speaker"] == "SYSTEM")
            assert(len(data[i]["turns"][j]["frames"]) == 1)
            slots = copy.deepcopy(data[i]["turns"][j]["frames"][0]["slots"])
            slots.sort(key = lambda x : -x["start"])


            # data[i]["turns"][j]["delex"] = delex
            target = ''
            belief = []
            for k in range(len(data[i]["turns"][j-1]["frames"])):
                for slot in data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"]:
                    belief += [[data[i]["turns"][j-1]["frames"][k]["service"].split("_")[0].lower(), slot, data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"][slot]]]
            belief.sort(key = lambda x : x[0] + " " + x[1])
            for k in range(len(belief)):
                belief[k][2].sort()
                belief[k][2] = belief[k][2][0]
            belief = [x[0] + " " + x[1] + " " + x[2] for x in belief]
            if len(belief) > 0:
                target += ('<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> ')
            else:
                target += ('<|belief|> <|endofbelief|> ')
            action = copy.deepcopy(data[i]["turns"][j]["frames"][0]["actions"])
            action.sort(key = lambda x : x["act"])


            # db result, use gold db results
            db_list = []
            for k in range(len(data[i]["turns"][j]["frames"])):
                if "gold_db" in data[i]["turns"][j]["frames"][k] and len(data[i]["turns"][j]["frames"][k]["gold_db"]) > 0:
                    # first_entry = data[i]["turns"][j]["frames"][k]["gold_db"]
                    db_list = []
                    for ind, first_entry in enumerate(data[i]["turns"][j]["frames"][k]["gold_db"]):
                        this_db_list = [domain + " " + x + " " + first_entry[x] for x in first_entry]
                        this_db_list = ["result " + str(ind)] + this_db_list
                        db_list.extend(this_db_list)

                if "service_results" in data[i]["turns"][j]["frames"][k] and len(data[i]["turns"][j]["frames"][k]["service_results"]) > 0:
                    num_results = len(data[i]["turns"][j]["frames"][k]["service_results"])
                    db_list = [str(num_results) + " results"] + db_list


            ### use the first value?
            action_list = []
            for x in action:
                if len(x["values"]) > 0:
                    action_list.append(domain + " " + x["act"].lower() + " " + x["slot"]+ " " + " ".join(x["values"]))
                else:
                    action_list.append(domain + " " + x["act"].lower() + " " + x["slot"])

            if len(db_list) > 0:
                target += ('<|dbresults|> ' + ", ".join(db_list) + ' <|endofdbresults|> ')
            else:
                # no need to retrieve new db
                target += ('<|dbresults|> <|nonewdb|> <|endofdbresults|> ')

            target += ('<|action|> ' + ", ".join(action_list) + ' <|endofaction|>')

            # # knowledge snippets '<|chitchat|>', '<|nochitchat|>', '<|knowledge|>', '<|endofknowledge|>'
            # if data[i]["turns"][j]["enrich"]:
            #     kg_text = " ".join(data[i]["turns"][j]["kg_snippets_text"])
            #     target += ('<|chitchat|> <|endofdecision|> <|knowledge|> ' + kg_text + ' <|endofknowledge|> ')
            #     this_response = data[i]["turns"][j]["enriched_utter"]
            # else:
            #     target += '<|nochitchat|> <|endofdecision|>'
            #     this_response = data[i]["turns"][j]["utterance"]
            # target += ('<|response|> ' + this_response + ' <|endofresponse|>')
            # target = target

            # data[i]["turns"][j]["target"] = target


            context = '<|context|> '
            for m in range(j):
                if m % 2 == 0:
                    context += '<|user|> '
                else:
                    context += '<|system|> '
                context += data[i]["turns"][m]["utterance"] + " "
            context += '<|endofcontext|> '

            # full_train_input = (context + target).replace("\n", " ").replace("\r", "")

            tillaction = context + target

            # # oracle test
            # context += ('<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> ').lower()
            # context += ('<|dbresults|> ' + ", ".join(db_list) + ' <|endofdbresults|> ')


            context = context.strip()

            data[i]["turns"][j]["context"] = context
            data[i]["turns"][j]["tillaction"] = tillaction

            # inlm += [full_train_input]
            # assert("\n" not in inlm[-1])
            # inlme += [(context).replace("\n", " ").replace("\r", "")]

        i += 1

    with open(tgt + "/processed_kg_select_" + args.data, "w") as f:
        json.dump(data, f, indent=1)

if __name__ == '__main__':
    random.seed(42)
    main()
