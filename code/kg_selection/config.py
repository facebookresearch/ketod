# Copyright (c) Meta Platforms, Inc. and its affiliates.

class parameters():

    prog_name = "retriever"
    # using original_data_unitable
    root_path = "/data/users/zhiyuchen/todkg_dataset/runs/kg_select/"
    output_path = "/data/users/zhiyuchen/outputs/"
    cache_dir = "/data/users/zhiyuchen/tmp/"

    model_save_name = "kg_select_bert_base_model2_new"

    train_file = root_path + "processed_kg_select_train_final.json"
    valid_file = root_path + "processed_kg_select_dev_final.json"

    simpletod_path = "/data/users/zhiyuchen/todkg_dataset/runs/model2_new/"
    # # test_file = root_path + "dataset/test.json"
    # # test_file = root_path + "dataset/train.json"
    test_file = simpletod_path + "test_all_inter.json"
    # test_file = root_path + "processed_kg_select_dev_final.json"

    # model choice: bert, roberta, albert
    pretrained_model = "bert"
    model_size = "bert-base-cased"

    # pretrained_model = "roberta"
    # model_size = "roberta-large"

    # train or test
    device = "cuda"
    mode = "train"
    resume_model_path = ""
    saved_model_path = output_path + "kg_select_bert_base_20210817231802/saved_model/loads/5/model.pt"
    build_summary = False

    option = "rand"
    neg_rate = 3
    topn = 3

    # threshold for select snippets
    thresh = 0.5
    tillaction_gold = False
    generate_all = True
    if_fill_train = False
    generate_all_neg_max = 30

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512
    dropout_rate = 0.1

    batch_size = 16
    batch_size_test = 16
    epoch = 300
    learning_rate = 3e-5

    report = 300
    report_loss = 100
