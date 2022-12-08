# -*- encoding: utf-8 -*-
import time
import logging
import torch
import numpy as np
from torch.backends import cudnn
import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='the huggingface model name')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--repetition', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=20)

    args = parser.parse_args()
    print(111)
    logger.info(args)
    # print(args.model_name)
    # print(args.batch_size)
    # print(args.repetition)
    # print(args.max_length)
    if args.model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    elif args.model_name == "gpt2-medium":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    elif args.model_name == "bert-large-uncased":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertForMaskedLM.from_pretrained("bert-large-uncased")
    elif args.model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    text = "Replace me by any text you'd like. Replace me by any text you'd like. I'm a good boy. " \
           "The following table is a comparison between Hugging Face and model. Please just do everythin you want to " \
           "do. "

    # batch_size_list = [1, 4, 8, 16, 32, 100, 200, 400]
    batch_size_list = [64]
    max_len_list = [10, 20, 40]
    ans = []
    for bsz in batch_size_list:
        for len in max_len_list:
            inputs = [text for _ in range(bsz)]
            start = time.time()
            encoded_input = tokenizer.batch_encode_plus(inputs,
                                                        add_special_tokens=True,
                                                        max_length=len,
                                                        truncation=True,
                                                        padding=True,
                                                        return_tensors='pt',
                                                        return_attention_mask=True,
                                                        return_token_type_ids=True,
                                                        )
            end = time.time()
            print("cost time {}".format(end-start))
            input_ids = encoded_input.get("input_ids").cuda()
            attention_mask = encoded_input.get("attention_mask").cuda()
            token_type_ids = encoded_input.get("token_type_ids").cuda()
            model.cuda()
            print('warm up ...\n')
            with torch.no_grad():
                for _ in range(100):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            torch.cuda.synchronize()
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            repetitions = args.repetition
            timings = np.zeros((repetitions, 1))
            print('testing ...\n')
            with torch.no_grad():
                for rep in tqdm.tqdm(range(repetitions)):
                    starter.record()
                    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    ender.record()
                    torch.cuda.synchronize()  # 等待GPU任务完成
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time

            avg = timings.sum() / repetitions
            ans.append(avg)
            print('\nmodel={}, batch_size={}, max_len={}, avg={}\n'.format(args.model_name, bsz, len, avg))
    print('\nans is {}\n'.format(ans))