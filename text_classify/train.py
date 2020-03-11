# coding=utf8

import csv
import numpy as np
import argparse
import torch
from torch.utils.data import dataloader
from torch.nn import functional as F

from tqdm import tqdm
from text_classify.model import TextCNN
from text_classify.tokenizer import jieba_tokenizer
from text_classify.utils import file_helper
from text_classify import dataset
from text_classify import metrics


class Trainer(object):
    def __init__(self, args):
        if args.emb_opt == 'rand':
            self.tokenizer = jieba_tokenizer.JiebaTokenizer(args.train_path, args.test_path, None)
        else:
            self.tokenizer = jieba_tokenizer.JiebaTokenizer(None, None, args.emb_file)
        if self.tokenizer.dim is not None:
            args.emb_dim = self.tokenizer.dim
            args.vocab_size = len(self.tokenizer.itos)
            if self.tokenizer.vecs is not None:
                args.weights = torch.from_numpy(np.array(self.tokenizer.vecs, dtype=np.float))
            else:
                args.weights = None
        self.model = TextCNN(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self, train_path, test_path, epochs, model_path, lr, batch_size):
        train_loader, label_dict = self.get_data_loader(train_path, batch_size=batch_size)
        test_loader, _ = self.get_data_loader(test_path, batch_size, label_dict)
        optimizer = torch.optim.SGD([
            {"params": self.model.parameters(), "lr": lr}
        ])
        scheduler = torch.optim.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=1000)
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
        texts, labels, label_dict = self._parse_file(f_path, label_dict)
        ds = dataset.RawDataSet(self.tokenizer, texts, labels)
        d_loader = dataloader.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=lambda x: dataset.fix_padding(x))
        return d_loader, label_dict

    @staticmethod
    def _parse_file(f_path, label_dict):
        texts = []
        raw_labels = []
        with open(f_path) as in_:
            reader = csv.reader(in_, delimiter="\t")
            next(reader)
            for fields in reader:
                if len(fields) >= 3:
                    raw_labels.append(fields[1])
                    texts.append(fields[2].strip())
        if label_dict is None:
            label_dict = dict([(elem, i) for i, elem in enumerate(sorted(set(raw_labels)))])
        Y = [label_dict[raw_label] for raw_label in raw_labels]
        return texts, Y, label_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=file_helper.get_data_file("train.tsv"))
    parser.add_argument("--test_path", type=str, default=file_helper.get_data_file("test.tsv"))
    parser.add_argument("--model_path", type=str, default=file_helper.get_data_file("model_data/cnn_model.cpk"))

    parser.add_argument("--emb_file", type=str, default=file_helper.get_data_file("model_data/sgns.zhihu.word"))
    parser.add_argument("--emb_opt", type=str, default="not_rand", choices=["rand", "not_rand"])
    parser.add_argument("--emb_freeze", type=bool, default=True,
                        help="option to freeze word vector or not for pretrained embeddings")
    parser.add_argument("--emb_dim", type=int, default=300, help="dimension of embedding layer")

    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--filter_sizes", type=str, default="2,3,4,5,6")
    parser.add_argument("--out_channel", default=64, type=int)
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
