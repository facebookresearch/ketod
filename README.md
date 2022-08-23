# The KETOD dataset

This repo contains the dataset from NAACL 2022 paper "KETOD: Knowledge-Enriched Task-Oriented Dialogue"
<https://arxiv.org/abs/2205.05589>

## Dataset Generation
KETOD is built upon the google SGD dataset. Here we release our knowledge-enriched utterances annotations and the script to generate the final dataset. 

1. Go to <https://github.com/google-research-datasets/dstc8-schema-guided-dialogue> to download the SGD dataset. 
2. Unzip ketod_release.zip and put with the SGD dataset in the same directory. 
3. Edit the main entry of gen_ketod_data.py to set up your own data paths. 
4. Run 'python gen_ketod_data.py' to generate the full KETOD dataset. 


## Dataset Format

Each entry of the data is one dialogue. It has the following fields:
```
"dialogue_id": unique id of the dialogue.

"turns": the list of dialogue turns. Besides the original fields in the SGD dataset, if it is an enriched turn, then we have the following additional fields:
    {
      "enrich": True. For turns without chitchat enrichment, this field is False. 
      "entity_query": The entity query we use to do knowledge retrieval.
      "enriched_utter": The utterance enriched with chitchat. Another field 'utterance' is the original response in the SGD dataset.
      "kg_snippets": the index of the ground truth knowledge snippets
      "kg_snippets_text": the content of the ground truth knowledge snippets
    }
  
"dialog_query": all the entity queries we use to do knowledge retrieval in this dialog

"entity_passages": all the wikipedia passages retrieved in this dialog

"entity_passage_sents": all the wikipedia passages retrieved in this dialog, breaked into snippets associated with index numbers
```

## Citation
If you find this project useful, please cite it using the following format

```
@inproceedings{DBLP:conf/naacl/ChenLMSCW22,
  author    = {Zhiyu Chen and
               Bing Liu and
               Seungwhan Moon and
               Chinnadhurai Sankar and
               Paul A. Crook and
               William Yang Wang},
  editor    = {Marine Carpuat and
               Marie{-}Catherine de Marneffe and
               Iv{\'{a}}n Vladimir Meza Ru{\'{\i}}z},
  title     = {{KETOD:} Knowledge-Enriched Task-Oriented Dialogue},
  booktitle = {Findings of the Association for Computational Linguistics: {NAACL}
               2022, Seattle, WA, United States, July 10-15, 2022},
  pages     = {2581--2593},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://doi.org/10.18653/v1/2022.findings-naacl.197},
  doi       = {10.18653/v1/2022.findings-naacl.197},
  timestamp = {Mon, 01 Aug 2022 16:27:57 +0200},
  biburl    = {https://dblp.org/rec/conf/naacl/ChenLMSCW22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
KETOD is released under MIT license, see [LICENSE](https://github.com/facebookresearch/ketod/blob/main/LICENSE) for details.
