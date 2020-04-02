# coding=utf8

import re
import math
import numpy as np
import jieba
import pandas as pd
import fasttext
import argparse
from text_classify.utils import file_helper

"""
-output model_cooking -autotune-validation cooking.valid
"""

money_pat = re.compile(r"[\d\.]+[亿万千百十个元]+")
time_pat_arr = [
    re.compile(r"[\d]+[年月日天小时]+"),
]

label_prefix = "__label__"
stop_words = list('.!?,\'/()，。《》|？（）【】 \t\n的上下“”了： ！、在我:也＂｜；❤') + ["建行",  "支行", "银行", "浙商", "中信"]


def prepare_data(f_path, train_path, val_path):
    titles, labels, _ = read_excel(f_path, title_key="信息标题", label_key="情感", link_key="链接")
    samples = _make_samples(titles, labels)
    np.random.seed(5)
    perm = np.random.permutation(len(samples))
    train_size = math.ceil(0.9 * len(samples))
    train_samples = samples[perm[:train_size]]
    val_samples = samples[perm[train_size:]]
    _save_samples(train_samples, train_path)
    _save_samples(val_samples, val_path)


def _save_samples(samples, f_path):
    with open(f_path, "w") as out_:
        for sample in samples:
            out_.write(sample + "\n")


def read_excel(f_path, title_key="标题", label_key="信息性质", link_key="网址"):
    df = pd.read_excel(f_path)
    raw_title = df[title_key]
    labels = df[label_key]
    links = df[link_key]
    return raw_title.values, labels.values, links.values


def _make_samples(titles, labels):
    result = [""] * len(titles)
    for i, title in enumerate(titles):
        label = labels[i]
        sample_line = construct_sample(title, label)
        result[i] = sample_line
    result = np.array(result)
    return result


def _clean_text(text):
    clear_text = text
    clear_text = re.sub(money_pat, " ", clear_text)
    for time_pat in time_pat_arr:
        clear_text = re.sub(time_pat, " ", clear_text)
    return clear_text


def construct_sample(text, label):
    text = _clean_text(text)
    words = [word for word in jieba.cut(text) if word not in stop_words]
    # words = [word for word in jieba.cut(text)]
    line = " ".join(([label_prefix + label] + words))
    return line


def train_and_valid(train_path, valid_path, model_path):
    model = fasttext.train_supervised(input=train_path, lr=1, epoch=600, wordNgrams=1)
    result = model.test(valid_path)

    model.save_model(model_path)
    print(result)


def main(args):
    file_helper.mk_folder_for_file(args.model_path)
    train_path, valid_path = args.input_train, args.input_valid
    input_path = args.input
    if not file_helper.is_file_exists(train_path):
        prepare_data(input_path, train_path, valid_path)
    train_and_valid(train_path, valid_path, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=file_helper.get_data_file("bank_data/sentiment_classification(1).xlsx"))
    parser.add_argument("--input_train", type=str, default=file_helper.get_data_file("bank_data/train.csv"))
    parser.add_argument("--input_valid", type=str, default=file_helper.get_data_file("bank_data/valid.csv"))
    parser.add_argument("--model_path", type=str, default=file_helper.get_data_file("model_data/fasttext.cpk"))
    args = parser.parse_args()
    main(args)
