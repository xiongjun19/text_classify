# coding=utf8


import pandas as pd
from text_classify.utils import file_helper

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_excel(f_path, title_key="标题", label_key="信息性质", link_key="网址"):
    df = pd.read_excel(f_path)
    raw_title = df[title_key]
    labels = df[label_key]
    links = df[link_key]
    return raw_title.values, labels.values, links.values


if __name__ == "__main__":
    t_in_file = file_helper.get_data_file("bank_data/【最终】中信银行，整理数据，3.2-3.8，3575条.xlsx")
    texts, labels, links = read_excel(t_in_file)
    print(texts[:10])
    print(labels[:10])
    print(links[:10])
