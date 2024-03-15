import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import more_torch_functions as mtf
from sklearn import metrics

from torch.utils.data import DataLoader, TensorDataset

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

def train_model(x, y, model, criterion, optimizer, max_epoch=1):
    for _ in range(max_epoch):
        model.train()
        y_pred = model(x)
        
        loss = criterion(y_pred, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    return y_pred

def one_epoch(loader, model, criterion, optimizer, concat=False):
    y_pred = torch.Tensor([]) if concat else None
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        if model.training:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if concat:
            y_pred = torch.concat((y_pred, outputs.detach()))

    return y_pred

def train_model_batch(train_loader, model, criterion, optimizer, max_epoch=1):
    for epoch in range(max_epoch):
        model.train()
        y_pred = one_epoch(train_loader, model, criterion, optimizer, concat=epoch+1 == max_epoch)
    return y_pred

def cross_valid(X, Y, model, criterion, optimizer, skf, *hooks_data, batch_size=-1, **kw_train):
    if batch_size < 0:
        return _cross_valid(X, Y, model, criterion, optimizer, skf, *hooks_data, **kw_train)
    else:
        return _cross_valid_batch(X, Y, model, criterion, optimizer, skf, batch_size, *hooks_data, **kw_train)

def _cross_valid_batch(X, Y, model, criterion, optimizer, skf, batch_size, *hooks_data, **kw_train):
    for train_index, valid_index in skf.split(X, Y):
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = Y[train_index], Y[valid_index]

        train_dataset = TensorDataset(x_train.unsqueeze(1), y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_valid.unsqueeze(1), y_valid) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        mtf.reset_model(model)

        # TODO hook data

        y_pred = train_model_batch(train_loader, model, criterion, optimizer, **kw_train)
        y_pred_train = y_pred.detach().round()
        model.eval()
        y_pred_eval = one_epoch(val_loader, model, criterion, optimizer, concat=True).detach()
        yield y_pred_train, y_train, y_pred_eval, y_valid

def _cross_valid(X, Y, model, criterion, optimizer, skf, *hooks_data, **kw_train):
    for train_index, valid_index in skf.split(X, Y):
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = Y[train_index], Y[valid_index]

        mtf.reset_model(model)
        
        for hook in hooks_data:
            hook(y_train, y_valid)

        y_pred = train_model(x_train, y_train, model, criterion, optimizer, **kw_train)
        y_pred_train = y_pred.detach().round()
        model.eval()
        y_pred_eval = model(x_valid).detach()
        yield y_pred_train, y_train, y_pred_eval, y_valid

def combine_prompts(prompts, sep):
    plen = len(prompts)//2
    return '\n\t              '.join([f"{vprompt}{sep}{tprompt}" for vprompt, tprompt in zip(prompts[:plen], prompts[plen:])])