#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and its affiliates.
import json



def get_tf_idf_query_similarity(allDocs, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query doc
    return: cosine similarity between query and all docs
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words='english')
    docs_tfidf = vectorizer.fit_transform(allDocs)

    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()

    # print(cosineSimilarities)
    return cosineSimilarities



def tfidf_retrieve(json_in, json_out):
    '''
    tfidf retriever
    '''


    with open(json_in) as f:
        data = json.load(f)


    # take top ten
    all_recall = 0.0
    all_recall_3 = 0.0
    all_kg_chitchat = 0

    res = []
    for each_data in data:
        all_snippets = []
        for each_query in each_data["entity_passages_sents"]:
            for each_passage in each_data["entity_passages_sents"][each_query]:
                passage_title = each_passage[0]
                for each_snippet in each_passage[1:]:
                    all_snippets.append(each_snippet[1])




        for ind, turn in enumerate(each_data["turns"]):
            prev_user_turn_utter = each_data["turns"][ind-1]["utterance"]
            all_utter = prev_user_turn_utter + turn["utterance"]

            tfidf_sim_mat = get_tf_idf_query_similarity(all_snippets, all_utter)

            tfidf_dict = {}
            for ind, score in enumerate(tfidf_sim_mat):
                tfidf_dict[ind] = score

            sorted_dict = sorted(tfidf_dict.items(), key=lambda kv: kv[1], reverse=True)

            retrieved = []
            retrieved_text = []
            for i in range(3):
                retrieved.append(sorted_dict[i][0])

            if "enrich" in turn and turn["enrich"]:
                correct_3 = 0
                all_kg_chitchat += 1


                for tmp in retrieved:
                    retrieved_text.append(all_snippets[tmp])
                    if tmp in turn["kg_snippets"]:
                        correct_3 += 1

                # all_recall += (float(correct) / len(gold_inds))
                all_recall_3 += (float(correct_3) / len(turn["kg_snippets"]))

            turn["merge_retrieved"] = retrieved_text



        res.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    res_3 = all_recall_3 / all_kg_chitchat

    res = "Top 3: " + str(res_3) + "\n"

    print(res)



if __name__ == '__main__':

    root = "/data/users/zhiyuchen/todkg_dataset/"
    json_in = root + "processed/test_final.json"
    json_out = root + "processed/test_final_tfidf.json"
    tfidf_retrieve(json_in, json_out)
