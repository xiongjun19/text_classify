# coding=utf8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        if args.emb_opt == "rand":
            self.emb_layer = RandEmbedding(args.vocab_size, args.emb_dim, args.dropout)
        else:
            self.emb_layer = WordEmbeddings(args.vocab_size, args.emb_dim, args.weights,
                                            freeze=args.emb_freeze, dropout=args.dropout)
        filter_sizes = args.filter_sizes
        out_channel = args.out_channel
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(1, out_channel, (size, args.emb_dim)) for size in filter_sizes]
        )
        self.dropout = None
        if args.dropout > 0.:
            self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * out_channel, args.class_num)

    def forward(self, input_ids):
        x = self.emb_layer(input_ids)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_list]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.fc(x)
        return logits


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, dim, weights, freeze=True, max_pos=None, dropout=0.):
        super(WordEmbeddings, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, dim, _weight=weights.float())
        self.word_embedding.weight.requires_grad = not freeze
        self.pos_embedding = None
        if max_pos is not None:
            self.pos_embedding = nn.Embedding(max_pos, dim)
        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        words_embeddings = self.word_embedding(input_ids)
        embeddings = words_embeddings
        if self.pos_embedding is not None:
            seq_len = input_ids.size(1)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
            pos_emb = self.pos_embedding(pos_ids)
            embeddings += pos_emb
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return embeddings


class RandEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, max_pos=None, dropout=0.):
        super(RandEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = None
        if max_pos is not None and max_pos > 0:
            self.pos_embedding = nn.Embedding(max_pos, dim)
        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        words_embeddings = self.word_embedding(input_ids)
        embeddings = words_embeddings
        if self.pos_embedding is not None:
            seq_len = input_ids.size(1)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
            pos_emb = self.pos_embedding(pos_ids)
            embeddings += pos_emb
        if self.droput is not None:
            embeddings = self.dropout(embeddings)
        return embeddings


class BertClassfication(nn.Module):
    def __init__(self, args):
        super(BertClassfication, self).__init__()
        self.bert = BertModel.from_pretrained(os.getenv('BERT_BASE_CHINESE', 'bert-base-chinese'))
        self.hidden_dim = 256
        self.linear = nn.Linear(768, self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.linear_cls = nn.Linear(self.hidden_dim, args.class_num)
        self.finetuning = args.finetuning
        self.dropout = None
        if args.dropout > 0:
            self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        if self.training and self.finetuning:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
        sequence_output = encoded_layers[-1]
        first_token = sequence_output[:, 0]
        if self.dropout is not None:
            hidden = self.dropout(first_token)
        hidden = self.linear(hidden)
        hidden = self.activation(hidden)
        logits = self.linear_cls(hidden)
        return logits


class TextRCNN(nn.Module):
    def __init__(self, args):
        super(TextRCNN, self).__init__()
        if args.emb_opt == "rand":
            self.emb_layer = RandEmbedding(args.vocab_size, args.emb_dim, args.dropout)
        else:
            self.emb_layer = WordEmbeddings(args.vocab_size, args.emb_dim, args.weights,
                                            freeze=args.emb_freeze, dropout=args.dropout)
        filter_sizes = args.filter_sizes
        out_channel = args.out_channel
        self.rnn = nn.GRU(
            args.emb_dim, args.hidden_dim, num_layers=2, bias=True,
            batch_first=True, dropout=args.dropout, bidirectional=True)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(1, out_channel, (size, args.hidden_dim * 2)) for size in filter_sizes]
        )
        self.dropout = None
        if args.dropout > 0.:
            self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * out_channel, args.class_num)

    def forward(self, input_ids):
        x = self.emb_layer(input_ids)
        x, _ = self.rnn(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_list]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.fc(x)
        return logits
