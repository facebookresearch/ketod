# Copyright (c) Meta Platforms, Inc. and its affiliates.
# generate non-delex data

import json
import copy
import random
import argparse

root = "/data/users/zhiyuchen/todkg_dataset/runs/model1/"
tgt = "/data/users/zhiyuchen/todkg_dataset/runs/model1/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train_final.json", type=str, required=False, help="data file name")
    args = parser.parse_args()


    inlm = []
    inlm_gold_action = []
    inlm_gold_knowledge = []
    inlm_gold_decision = []
    inlme = []

    inlm_rand = []
    inlme_rand = []
    inlme_rand_kg = []

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
            delex = data[i]["turns"][j]["utterance"]


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
                    action_list.append(domain + " " + x["act"].lower() + " " + x["slot"] + " " + " ".join(x["values"]))
                else:
                    action_list.append(domain + " " + x["act"].lower() + " " + x["slot"])

            if len(db_list) > 0:
                target += ('<|dbresults|> ' + ", ".join(db_list) + ' <|endofdbresults|> ')
            else:
                # no need to retrieve new db
                target += ('<|dbresults|> <|nonewdb|> <|endofdbresults|> ')

            target += ('<|action|> ' + ", ".join(action_list) + ' <|endofaction|> ')
            target_rand = target[:]

            # knowledge snippets '<|chitchat|>', '<|nochitchat|>', '<|knowledge|>', '<|endofknowledge|>'
            # if data[i]["turns"][j]["enrich"]:
            #     kg_text = " ".join(data[i]["turns"][j]["kg_snippets_text"])
            #     target += ('<|knowledge|> ' + kg_text + ' <|endofknowledge|> <|chitchat|>')
            # else:
            #     this_kg_text_list = []
            #     for each_query in data[i]["turns"][j]["entity_passages_sents"]:
            #         for each_passage in data[i]["turns"][j]["entity_passages_sents"][each_query]:
            #             passage_title = each_passage[0]

            #             for each_snippet in each_passage[1:]:
            #                 if int(each_snippet[0]) in data[i]["turns"][j]["retrieved"]:
            #                     this_kg_text_list.append(passage_title + " " + each_snippet[1])


            this_kg_text_list = []
            # print(data[i].keys())
            for each_query in data[i]["entity_passages_sents"]:
                for each_passage in data[i]["entity_passages_sents"][each_query]:
                    passage_title = each_passage[0]

                    for each_snippet in each_passage[1:]:
                        this_kg_text_list.append(passage_title + " " + each_snippet[1])

            rand_snippets = random.choices(this_kg_text_list, k=3)
            rand_snippets_text = " ".join(rand_snippets)

            this_kg_text_list = data[i]["turns"][j]["merge_retrieved"]
            kg_text = " ".join(this_kg_text_list)
            if data[i]["turns"][j]["enrich"]:
                target += ('<|knowledge|> ' + kg_text + ' <|endofknowledge|> <|chitchat|> <|endofdecision|> ')
                target_rand += ('<|knowledge|> ' + rand_snippets_text + ' <|endofknowledge|> <|chitchat|> <|endofdecision|> ')
                this_response = data[i]["turns"][j]["enriched_utter"]
            else:
                target += ('<|knowledge|> ' + kg_text + ' <|endofknowledge|> <|nochitchat|> <|endofdecision|> ')
                target_rand += ('<|knowledge|> ' + rand_snippets_text + ' <|endofknowledge|> <|nochitchat|> <|endofdecision|> ')
                this_response = data[i]["turns"][j]["utterance"]


            target += ('<|response|> ' + this_response + ' <|endofresponse|>')
            target_rand += ('<|response|> ' + this_response + ' <|endofresponse|>')

            data[i]["turns"][j]["target"] = target


            context = '<|context|> '
            for m in range(j):
                if m % 2 == 0:
                    context += '<|user|> '
                else:
                    context += '<|system|> '

                if m % 2 == 1 and data[i]["turns"][m]["enrich"]:
                    context += data[i]["turns"][m]["enriched_utter"] + " "
                else:
                    context += data[i]["turns"][m]["utterance"] + " "
            context += '<|endofcontext|> '


            full_train_input = (context + target).replace("\n", " ").replace("\r", "")
            full_train_input_rand = (context + target_rand).replace("\n", " ").replace("\r", "")

            tillaction = (context + target.split('<|endofaction|>')[0].strip() + " <|endofaction|>").replace("\n", " ").replace("\r", "")
            tillkg = (context + target.split('<|endofknowledge|>')[0].strip() + " <|endofknowledge|>").replace("\n", " ").replace("\r", "")
            tilldecision = (context + target.split('<|endofdecision|>')[0].strip() + " <|endofdecision|>").replace("\n", " ").replace("\r", "")
            tillkg_rand = (context + target_rand.split('<|endofknowledge|>')[0].strip() + " <|endofknowledge|>").replace("\n", " ").replace("\r", "")

            # # oracle test
            # context += ('<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> ').lower()
            # context += ('<|dbresults|> ' + ", ".join(db_list) + ' <|endofdbresults|> ')


            context = context.strip()

            data[i]["turns"][j]["context"] = context
            data[i]["turns"][j]["tillaction"] = tillaction
            data[i]["turns"][j]["tilldecision"] = tilldecision
            data[i]["turns"][j]["tillkg"] = tillkg
            data[i]["turns"][j]["full_train"] = full_train_input
            inlm += [full_train_input]
            inlm_rand += [full_train_input_rand]

            inlm_gold_action += [tillaction]
            inlm_gold_knowledge += [tillkg]
            inlm_gold_decision += [tilldecision]
            inlme_rand_kg += [tillkg_rand]


            assert("\n" not in inlm[-1])
            inlme += [(context).replace("\n", " ").replace("\r", "")]

        i += 1

    with open(tgt + "/processed_model1_" + args.data, "w") as f:
        json.dump(data, f, indent=1)


    # random.shuffle(inlm)
    with open(tgt + "model1.lm.input."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #SimpleTOD
        f.write('\n'.join(inlm))
    with open(tgt + "model1.lm.rand.input."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #SimpleTOD
        f.write('\n'.join(inlm_rand))
    with open(tgt + "model1.lm.input.eval.goldaction."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #SimpleTOD
        f.write('\n'.join(inlm_gold_action))
    with open(tgt + "model1.lm.input.eval.goldkg."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #SimpleTOD
        f.write('\n'.join(inlm_gold_knowledge))
    with open(tgt + "model1.lm.rand.input.eval.goldkg."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #SimpleTOD
        f.write('\n'.join(inlme_rand_kg))
    with open(tgt + "model1.lm.input.eval.golddecision."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #SimpleTOD
        f.write('\n'.join(inlm_gold_decision))
    with open(tgt + "model1.lm.input.eval."+ args.data.split(".")[0] +".txt", "w", encoding='utf8') as f: #used as the input during evaluation of SimpleTOD and SimpleTOD extension
        f.write('\n'.join(inlme))

if __name__ == '__main__':
    random.seed(42)
    main()
