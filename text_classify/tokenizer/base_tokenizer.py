# coding=utf8


"""
本文件主要是实现对分词以及word 到词典的映射的封装
"""


class BaseTokenizer(object):

    def tokenize(self, sen):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        raise NotImplementedError
