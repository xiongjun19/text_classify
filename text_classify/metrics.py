# coding=utf8

import torch
from sklearn.metrics import classification


def calc_accuracy(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    pred_labels = torch.argmax(y_pred, dim=1)
    pred_labels = pred_labels.cpu().numpy()
    acc = classification.accuracy_score(y_true, pred_labels)
    return acc


def calc_f1(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    pred_labels = torch.argmax(y_pred, dim=1)
    pred_labels = pred_labels.cpu().numpy()
    if y_pred.size(1) > 2:
        return classification.f1_score(y_true, pred_labels, average="macro")
    return classification.f1_score(y_true, pred_labels)



