from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)
import torch
import numpy as np

def predict_val(model, device):
    candidate_list = ["positive", "neutral", "negative"]

    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open("/data/sentiment_generation/ACSA/vote_dataset/MAMS/MAMS_val.txt", "r") as f:
        file = f.readlines()
    train_data = []
    count = 0
    total = 0
    for line in file:
        total += 1
        # score_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']

        # target_list = ["For " + term.lower() + ", the sentiment is " + candi.lower() + " ." for candi in candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list1.append(score)

        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score)
        score_list = score_list2

        # target_list = ["The " + term.lower() + " category has a " + candi.lower() + " label ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list3.append(score)

        # target_list = ["The sentiment is " + candi.lower() + " for " + term.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list4.append(score)

        # target_list = ["The " + term.lower() + " is " + candi.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list5.append(score)

        # score_list = [(score_list1[i] + score_list2[i] + score_list3[i]) for i in range(0, len(score_list1))]
        predict = candidate_list[np.argmax(score_list)]
        if predict == golden_polarity:
            count += 1
            print(predict, golden_polarity, count/total, count, total)

    return count/total

def predict_test(model, device):
    candidate_list = ["positive", "neutral", "negative"]

    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open("/data/sentiment_generation/ACSA/vote_dataset/MAMS/MAMS_test.txt", "r") as f:
        file = f.readlines()
    train_data = []
    count = 0
    total = 0
    for line in file:
        total += 1
        # score_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']

        # target_list = ["For " + term.lower() + ", the sentiment is " + candi.lower() + " ." for candi in candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list1.append(score)

        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score)
        score_list = score_list2

        # target_list = ["The " + term.lower() + " category has a " + candi.lower() + " label ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list3.append(score)

        # target_list = ["The sentiment is " + candi.lower() + " for " + term.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list4.append(score)

        # target_list = ["The " + term.lower() + " is " + candi.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list5.append(score)

        # score_list = [(score_list1[i] + score_list2[i]) for i in range(0, len(score_list1))]
        predict = candidate_list[np.argmax(score_list)]
        if predict == golden_polarity:
            count += 1
            print(predict, golden_polarity, count/total, count, total)

    return count/total

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model2 = BartForConditionalGeneration.from_pretrained('/data/sentiment/models/MAMS/model2/checkpoint-3108-epoch-7').to(device)
# acc = predict_test(model2, device)