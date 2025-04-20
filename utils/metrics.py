from sklearn import metrics
import numpy as np


def compute_metrics(all_trues, all_scores, threshold):
    all_preds = (all_scores >= threshold)
    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    AUPR = metrics.average_precision_score(all_trues, all_scores)

    return acc, pre, rec, f1, AUC, AUPR


def print_metrics(data_type, metrics):
    """ Print the evaluation results """
    acc, pre, rec, f1, AUC, AUPR = metrics
    res = '\t'.join([
        '%s:' % data_type,
        'acc:%0.6f' % acc,
        'pre:%0.6f' % pre,
        'rec:%0.6f' % rec,
        'f1:%0.6f' % f1,
        'auc:%0.6f' % AUC,
        'aupr:%0.6f' % AUPR
    ])
    print(res)

def best_f1_thr(y_true, y_score):
    """ Calculate the best threshold with f1-score """
    best_thr = 0.5
    best_f1 = 0
    for thr in range(1, 100):
        thr /= 100
        acc, pre, rec, f1, AUC, AUPR = compute_metrics(y_true, y_score, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1