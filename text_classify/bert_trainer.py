# coding=utf8

import os
import math
import numpy as np
import argparse
import torch
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import dataloader
from torch.nn import functional as F

from tqdm import tqdm
from text_classify.model import BertClassfication
from text_classify.utils import file_helper
from text_classify import dataset
from text_classify import metrics


class Trainer(object):
    def __init__(self, args):

        self.tokenizer = BertTokenizer.from_pretrained(os.getenv("BERT_BASE_CHINESE_VOCAB", "bert-base-chinese"),
                                                       do_lower_case=True)(None, None, args.emb_file)
        self.model = BertClassfication(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self, train_path, test_path, epochs, model_path, lr, batch_size):
        train_loader, test_loader, label_dict = self.get_data_loader(train_path, batch_size=batch_size)
        optimizer = torch.optim.SGD([
            {"params": self.model.parameters(), "lr": lr}
        ])
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=2e-6, max_lr=2e-4, step_size_up=1000)
        criterion = F.cross_entropy
        for epoch in tqdm(range(epochs)):
            self.train_epoch(train_loader, test_loader, optimizer, scheduler, criterion,  epoch)
            print(f"following is test metrics at epoch {epoch}")
            acc, f1_score = self.evaluate(test_loader)
            print(f"accuracy of the model is: {acc}")
            print(f"f1 score is: {f1_score}")
            print(f"following is train metrics at epoch {epoch}")
            acc, f1_score = self.evaluate(train_loader)
            print(f"accuracy of the model is: {acc}")
            print(f"f1 score is: {f1_score}")
            torch.save(self.model.state_dict(), model_path)

    def evaluate(self, _loader):
        self.model.eval()
        acc_arr = []
        f1_arr = []
        with torch.no_grad():
            for batch in _loader:
                x, y, _, _ = batch
                x = x.to(self.device)
                logits = self.model(x)
                acc = metrics.calc_accuracy(y, logits)
                f1 = metrics.calc_f1(y, logits)
                acc_arr.append(acc)
                f1_arr.append(f1)
        acc_score = sum(acc_arr) / len(acc_arr) if len(acc_arr) > 0 else -1
        f1_score = sum(f1_arr) / len(f1_arr) if len(f1_arr) > 0 else -1
        return acc_score, f1_score

    def train_epoch(self, train_loader, test_loader, optmizer, scheduler, criterion,  epoch):
        self.model.train()
        for step, batch in enumerate(train_loader):
            self.model.zero_grad()
            x, y, _, _ = batch
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            scheduler.step()
            optmizer.step()
            if step % 10 == 0:
                print(f"training loss is: {loss.cpu().item()} @step:{step}, @epoch: {epoch}")

    def get_data_loader(self, f_path, batch_size, label_dict=None):
        np.random.seed(5)
        texts, labels, label_dict = self._parse_file(f_path, label_dict)
        perm = np.random.permutation(len(texts))
        train_size = math.ceil(0.9 * len(texts))
        train_texts = texts[perm[:train_size]]
        train_labels = labels[perm[:train_size]]
        test_texts = texts[perm[train_size:]]
        test_labels = labels[perm[train_size:]]
        train_ds = dataset.RawDataSet(self.tokenizer, train_texts, train_labels)
        test_ds = dataset.RawDataSet(self.tokenizer, test_texts, test_labels)
        train_loader = dataloader.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                             collate_fn=lambda x: dataset.fix_padding(x))
        test_loader = dataloader.DataLoader(test_ds, batch_size=batch_size, shuffle=True,
                                            collate_fn=lambda x: dataset.fix_padding(x))
        return train_loader, test_loader, label_dict

    def _parse_file(self, f_path, label_dict):
        texts, raw_labels, links = self.read_excel(f_path)
        if label_dict is None:
            label_dict = dict([(elem, i) for i, elem in enumerate(sorted(set(raw_labels)))])
        Y = [label_dict[raw_label] for raw_label in raw_labels]
        return texts, np.array(Y), label_dict

    @staticmethod
    def read_excel(f_path, title_key="标题", label_key="信息性质", link_key="网址"):
        df = pd.read_excel(f_path)
        raw_title = df[title_key]
        labels = df[label_key]
        links = df[link_key]
        return raw_title.values, labels.values, links.values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=file_helper.get_data_file("train.tsv"))
    parser.add_argument("--test_path", type=str, default=file_helper.get_data_file("test.tsv"))
    parser.add_argument("--model_path", type=str, default=file_helper.get_data_file("model_data/cnn_model.cpk"))
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--class_num", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="learing rate")
    parser.add_argument("--epochs", default=30, type=int, help="epochs")
    parser.add_argument("--eval_steps", default=10, type=int, help="interval to evaluate")
    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()
    filter_sizes = list(map(int, args.filter_sizes.split(",")))
    args.filter_sizes = filter_sizes
    return args


def main(args):
    trainer = Trainer(args)
    train_path = args.train_path
    test_path = args.test_path
    epochs = args.epochs
    model_path = args.model_path
    lr = args.lr
    batch_size = args.batch_size
    trainer.train(train_path, test_path, epochs, model_path, lr, batch_size)


if __name__ == "__main__":
    t_args = parse_args()
    main(t_args)
