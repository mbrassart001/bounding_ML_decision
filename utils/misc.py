import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from sklearn import metrics

sys.path.append(os.path.dirname(__file__))

import more_torch_functions as mtf
from torch.utils.data import DataLoader, TensorDataset

def set_current_epoch(epoch):
    global glob_epoch
    glob_epoch = epoch

def get_current_epoch():
    global glob_epoch
    return glob_epoch

def reset_current_epoch():
    global glob_epoch
    glob_epoch = -2

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

def one_epoch(loader, model, criterion=None, optimizer=None, concat=False):
    y_pred = torch.Tensor([]) if concat else None
    for inputs, labels in loader:
        if model.training:
            model.zero_grad()
        outputs = model(inputs)
        if model.training:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if concat:
            y_pred = torch.concat((y_pred, outputs.detach()))

    return y_pred

def train_model_batch(train_loader, model, criterion, optimizer, max_epoch=1):
    model.train()
    for epoch in range(max_epoch):
        set_current_epoch(epoch)
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
        reset_current_epoch()
        model.eval()
        y_pred_valid = one_epoch(val_loader, model, concat=True).detach()
        yield y_pred_train, y_train, y_pred_valid, y_valid

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

class Figures(widgets.HBox):
    def __init__(self, x, y, labels=None, options=None, **kwargs):
        super().__init__()
 
        self.x = x
        self.y = y
        self.l = labels if labels is not None else [i+1 for i in range(self.y[0].shape[0])]

        subplots_kw = ['nrows', 'ncols', 'sharex', 'sharey', 'subplot_kw', 'gridspec_kw']
        self.subplots_kwargs = {k: kwargs.pop(k) for k in subplots_kw if k in kwargs}
        self.fig_kw = kwargs

        self.widgets_init(options)
        self.figure_init()

    def widgets_init(self, options):
        self.output = widgets.Output()
        toggle_buttons = widgets.ToggleButtons(
            options=self.l + ['all'],
            disabled=False,
            value='all',
        )
        self.toggle_buttons = toggle_buttons
        toggle_buttons.observe(self.update)
        controls = toggle_buttons

        if options is not None:
            radio_buttons = widgets.RadioButtons(
                options=options,
                disable= False,
                value=options[0],
            )
            radio_buttons.observe(self.change_orientation)
            controls = widgets.VBox([radio_buttons, toggle_buttons])

        self.children = [controls, self.output]

    def lines_init(self):
        self.lines = []
        for ax, y in zip(self.axs.flat, self.y):
            for y_values, label in zip(y, self.l):
                self.lines.append(ax.plot(self.x, y_values, **self.fig_kw, label=label)[0])
            ax.grid(True)

    def figure_init(self):
        self.curve_per_plot = len(self.l)
        self.plot_autocenter(True)

        with self.output:
            self.fig, self.axs = plt.subplots(**self.subplots_kwargs, constrained_layout=True, figsize=(5, 3.5))

        self.lines_init()

        self.fig.canvas.toolbar_position = 'bottom'
        self.fig.canvas.header_visible = False

    def visible_legend(self, visible: bool):
        for ax in self.axs.flat:
            legend = ax.legend_
            if legend is not None:
                legend.set_visible(visible)

    def _legend(self):
        col = self.legend_args.get('col')
        row = self.legend_args.get('row')
        kwargs = self.legend_args.get('kwargs')
        if row is not None and col is not None:
            axs = [self.axs[row,col]]
        elif row is not None:
            axs = self.axs[row,:]
        elif col is not None:
            axs = self.axs[:,col]
        else:
            axs = self.axs.flat
        
        for ax in axs:
            ax.legend(**kwargs)

    def legend(self, col=None, row=None, **kwargs):
        self.legend_args = {'col': col, 'row': row, 'kwargs': kwargs}
        self._legend()

    def annotate(self, cols: int, rows: int, pad=5):
        self.annotate_cols(cols, pad)
        self.annotate_rows(rows, pad)

    def annotate_cols(self, cols, pad=5):
        for ax, col in zip(self.axs[0], cols):
            ax.annotate(
                col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline'
            )

    def annotate_rows(self, rows, pad=5):
        for ax, row in zip(self.axs[:,0], rows):
            ax.annotate(
                row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center'
            )

    def plot_autocenter(self, center: bool):
        self.center_plot = center

    def update(self, change):
        try:
            new_index = change.new.get('index')
        except AttributeError:
            pass
        else:
            if new_index is None:
                return

            elif new_index >= self.curve_per_plot:
                for line in self.lines:
                    line.set(**self.fig_kw)
                for ax, y in zip(self.axs.flat, self.y):
                    min_data, max_data = y.min(), y.max()

                    if self.center_plot:
                        border_space = (max_data - min_data)/10
                        ax.set_ylim(min_data - border_space, max_data + border_space)
                self.visible_legend(True)

            else:
                for line in self.lines:
                    line.set(linestyle='None', marker='None')
                for next_line in self.lines[new_index::self.curve_per_plot]:
                    next_line.set(**self.fig_kw)

                for ax, y in zip(self.axs.flat, self.y):
                    data = y[new_index]
                    min_data, max_data = min(data), max(data)

                    if self.center_plot:
                        border_space = (max_data - min_data)/10
                        ax.set_ylim(min_data - border_space, max_data + border_space)
                self.visible_legend(False)
            
            self.fig.canvas.draw()
    
    def change_orientation(self, change):
        if change.get('name') == 'value':
            self.x, self.l = self.l, self.x
            self.y = [y.T for y in self.y]
            self.curve_per_plot = len(self.l)
            for ax in self.axs.flat:
                ax.cla()
            self.lines_init()
            self._legend()
            self.visible_legend(True)
            self.toggle_buttons.options = self.l + ['all']
            self.toggle_buttons.value = 'all'