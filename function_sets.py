import json
import csv
from nltk.stem import PorterStemmer as porter
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import torch
import random

def read_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def write_to_json(filename, data):

    json_string = json.dumps(data)
    with open(filename, 'w') as outfile:
        outfile.write(json_string)

        outfile.close()


def json_to_csv(json_file, csv_file):
    # 1.分别 读，创建文件
    json_fp = open(json_file, "r", encoding='utf-8')
    csv_fp = open(csv_file, "w", encoding='utf-8', newline='')

    # 2.提出表头和表的内容
    data_list = json.load(json_fp)
    sheet_title = data_list[0].keys()
    # sheet_title = {"姓名","年龄"}  # 将表头改为中文
    sheet_data = []
    for data in data_list:
        sheet_data.append(data.values())

    # 3.csv 写入器
    writer = csv.writer(csv_fp)

    # 4.写入表头
    writer.writerow(sheet_title)

    # 5.写入内容
    writer.writerows(sheet_data)

    # 6.关闭两个文件
    json_fp.close()
    csv_fp.close()



def stemmer(stem_text):
    """The function to apply stemming"""
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def tokenize(bert_tokenizer, texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.

    Args:
        texts (List[str]): List of text data

    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """
    trans_tokens = [bert_tokenizer.tokenize(text) for text in texts]
    trans_ids = [bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in trans_tokens]
    decode_text = [bert_tokenizer.decode(ids) for ids in trans_ids]
    texts = np.array(decode_text)

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    word2idx['['] = 2
    word2idx['SEP'] = 3
    word2idx[']'] = 4


    # Building our vocab from the corpus starting from index 2
    idx = 5
    for sent in texts:
        tokenized_sent = word_tokenize(sent)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len


def encode(tokenized_texts, word2idx, max_len):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)

    return np.array(input_ids)

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

