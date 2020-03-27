# coding=utf8

import argparse
import pandas as pd
import torch
from torch.utils.data import dataloader
import numpy as np
from text_classify.tokenizer import jieba_tokenizer
from text_classify.model import TextCNN
from text_classify import dataset
from text_classify import metrics


class Predictor(object):
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
        self.model.load_state_dict(args.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_file(self, f_path):
        texts, labels, label_dict = self._parse_file(f_path, None)
        print("the label dict is as following: ")
        print(label_dict)
        ds = dataset.RawDataSet(self.tokenizer, texts, labels)
        d_loader = dataloader.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=lambda x: dataset.fix_padding(x))
        acc_score, f1_score, cm = self.evaluate(d_loader)
        print(f"following is test metrics at epoch {epoch}")
        print(f"accuracy of the model is: {acc}")
        print(f"f1 score is: {f1_score}")
        print(f"confusion matrix is: {cm}")

    def evaluate(self, _loader):
        self.model.eval()
        acc_arr = []
        f1_arr = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in _loader:
                x, y, _, _ = batch
                x = x.to(self.device)
                logits = self.model(x)
                acc = metrics.calc_accuracy(y, logits)
                f1 = metrics.calc_f1(y, logits)
                y_true.append(y)
                y_pred.append(logits)
                acc_arr.append(acc)
                f1_arr.append(f1)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        cm = metrics.calc_cm(y_true, y_pred)
        acc_score = sum(acc_arr) / len(acc_arr) if len(acc_arr) > 0 else -1
        f1_score = sum(f1_arr) / len(f1_arr) if len(f1_arr) > 0 else -1
        return acc_score, f1_score, cm

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


def main():
    args = parse_args()
    predictor = Predictor(args)
    predictor.predict_file(args.train_path)


if __name__ == "__main__":
    main()