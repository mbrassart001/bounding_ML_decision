import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import more_torch_functions as mtf
from sklearn import metrics

def cm(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[False, True])
    return cm_display

def plot_cm(y_true, y_pred):
    cm_display = cm(y_true, y_pred)
    _, ax = plt.subplots(1, 1, figsize=(4,8))
    cm_display.plot(ax=ax, colorbar=False)

def plot_combine_cm(cms, titles=None):
    n = len(cms)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 8))
    if titles:
        for ax, cm, title in zip(axs, cms, titles):
            cm.plot(ax=ax, colorbar=False)
            ax.set_title(title)
    else:
        for ax, cm in zip(axs, cms):
            cm.plot(ax=ax, colorbar=False)
    fig.tight_layout()

def cov_score(y_true, y_pred):
    labels = np.unique(y_true)
    scores = {}

    for label in labels:
        indices_true = np.where(y_true == label)[0]
        indices_pred = np.where(y_pred == label)[0]
        scores[label] = len(np.intersect1d(indices_true, indices_pred))/len(indices_true)

    return scores

def train_model(x, y, model, loss_fn, optimizer, max_epoch):
    for _ in range(max_epoch):
        model.train()
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    return y_pred

def cross_valid(X, Y, model, loss_fn, optimizer, skf, *hooks_data, **kw_train):
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        mtf.reset_model(model)
        
        for hook in hooks_data:
            hook(y_train, y_test)

        y_pred = train_model(x_train, y_train, model, loss_fn, optimizer, **kw_train)
        y_pred_train = y_pred.detach().round()
        model.eval()
        y_pred_eval = model(x_test).detach()
        yield y_pred_train, y_train, y_pred_eval, y_test

def combine_prompts(prompts, sep):
    plen = len(prompts)//2
    return '\n\t              '.join([f"{vprompt}{sep}{tprompt}" for vprompt, tprompt in zip(prompts[:plen], prompts[plen:])])