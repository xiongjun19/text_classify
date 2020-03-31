# coding=utf8


import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class DatasetMixin(Dataset):
    def __init__(self):
        super(DatasetMixin, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()
        if isinstance(index, slice):
            begin, end, step = index.indices(len(self))
            return [self.get_example(i) for i in range(begin, end, step)]
        if isinstance(index, list):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def get_example(self, i):
        raise NotImplementedError


class RawDataSet(DatasetMixin):
    def __init__(self, tokenizer, texts, labels=None):
        """
        :param tokenizer:
        :param texts:  这里的texts 是原始文本
        :param labels: 这里的labels
        """
        super(RawDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def get_example(self, i):
        text = self.texts[i]
        label = self.labels[i]
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            tokens = ["[UNK]"]
        tok_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(tokens)
        return tok_ids, label, text, seq_len


def dynamic_pad(batch, upper_bound=512):
    """
    本函数对所有的sentence pad 到batch里面最长的sequence
    :param batch: [[tok_ids, label, text, seq_len]]
    :param upper_bound: 最多允许的seq 长度
    :return: [[tok_ids]], [[label]], [[text]], [[seq_len]]
    """
    tok_ids_arr, labels, text, seq_lens = zip(*batch)
    max_len = np.array(seq_lens).max()
    max_len = min(max_len, upper_bound)
    return _pad_and_convert(tok_ids_arr, labels, text, seq_lens, max_len)


def fix_padding(batch, upper_bound=512):
    """
    本函数将sentence  pad 到upper_bound
    :param batch:
    :param upper_bound:
    :return:
    """
    tok_ids_arr, labels, text, seq_lens = zip(*batch)
    max_len = upper_bound
    return _pad_and_convert(tok_ids_arr, labels, text, seq_lens, max_len)


def _pad_and_convert(tok_ids_arr, labels, text, seq_lens, max_len):
    # import ipdb; ipdb.set_trace()
    padded_tok_arr = []
    for i, tok_ids in enumerate(tok_ids_arr):
        if len(tok_ids) >= max_len:
            padded_tok_arr.append(tok_ids[:max_len])
        else:
            tmp_arr = tok_ids + (max_len - len(tok_ids)) * [0]
            padded_tok_arr.append(tmp_arr)
    return torch.LongTensor(padded_tok_arr), torch.LongTensor(labels), text, seq_lens


class BertDataSet(DatasetMixin):
    START_TAG = '[CLS]'
    STOP_TAG = '[SEP]'

    def __init__(self, tokenizer, texts, labels=None):
        """
        :param tokenizer:
        :param texts:  这里的texts 是原始文本
        :param labels: 这里的labels
        """
        super(BertDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def get_example(self, i):
        text = self.texts[i]
        label = self.labels[i]
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            tokens = ["[UNK]"]
        tokens = [self.START_TAG] + tokens + [self.STOP_TAG]
        tok_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(tokens)
        return tok_ids, label, text, seq_len
