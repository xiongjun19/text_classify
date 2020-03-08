# coding=utf8

import csv
import jieba
import numpy as np
from text_classify.tokenizer.base_tokenizer import BaseTokenizer


class JiebaTokenizer(BaseTokenizer):
    reserve_toks = ["[PAD]", "[UNK]"]

    def __init__(self, train_file, dev_file, vec_file=None):
        super(JiebaTokenizer, self).__init__()
        self.vecs = None
        self.dim = None
        if vec_file is not None:
            self.itos, self.vecs, self.dim = self.load_vec(vec_file)
        else:
            self.itos = self._construct_dict(train_file, dev_file)
        for i, w in enumerate(self.reserve_toks):
            self.itos[i] = w
        self.stoi = dict([(v, k) for k, v in self.itos.items()])

    def tokenize(self, sen):
        words = list(jieba.cut(sen))
        return words

    def convert_tokens_to_ids(self, tokens):
        result = [0] * len(tokens)
        for i, word in enumerate(tokens):
            idx = self.stoi.get(word, '[UNK]')
            result[i] = idx
        return result

    def _construct_dict(self, train_file, dev_file):
        train_words = self._read_words(train_file)
        dev_words = self._read_words(dev_file)
        words = train_words + dev_words
        itos = dict()
        for i, word in enumerate(sorted(words)):
            itos[i + len(self.reserve_toks)] = word
        return itos

    def _read_words(self, f_path):
        _set = set()
        with open(f_path) as in_:
            for line in in_:
                words = jieba.cut(line.strip())
                _set.update(words)
        return _set

    @classmethod
    def load_vec(cls, vec_file):
        index_to_word = {}
        weights = []
        with open(vec_file) as in_:
            reader = csv.reader(in_, delimiter=" ", quotechar=None)
            next(reader)
            for i, fields in enumerate(reader):
                word = fields[0]
                vec = np.array(list(map(float, fields[1:-1])))
                index_to_word[i + len(cls.reserve_toks)] = word
                weights.append(vec)
        dim = len(weights[0])

        result = [np.zeros(dim, dtype=float), np.random.normal(size=dim)]
        result.extend(weights)
        return index_to_word, result, dim


if __name__ == "__main__":
    from text_classify.utils import file_helper
    test_file = file_helper.get_data_file("model_data/sgns.zhihu.word")
    tokenizer = JiebaTokenizer(None, None, test_file)
    test_sen = "我是中国人"
    words = tokenizer.tokenize(test_sen)
    print(words)
    ids = tokenizer.convert_tokens_to_ids(words)
    print(ids)
