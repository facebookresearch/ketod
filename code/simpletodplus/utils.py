# Copyright (c) Meta Platforms, Inc. and its affiliates.
def write_log(log_file, s):
    print(s)
    with open(log_file, "a") as f:
        f.write(s + "\n")


def get_kg_snippets_dict(data, snippets_key_name="entity_passages_sents"):
    """
    get the kg_snippets as a dict to its ids
    """

    # make the dict
    all_kg_snippets_dict = {}
    for each_query in data[snippets_key_name]:
        for each_passage in data[snippets_key_name][each_query]:
            passage_title = each_passage[0]

            for each_snippet in each_passage[1:]:
                assert int(each_snippet[0]) not in all_kg_snippets_dict
                all_kg_snippets_dict[int(each_snippet[0])] = (
                    passage_title + " " + each_snippet[1]
                )

    return all_kg_snippets_dict
