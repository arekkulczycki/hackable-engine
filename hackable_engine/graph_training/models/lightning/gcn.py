# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
from torch.nn import Dropout, Linear, ReLU
import torch_geometric
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool
# this import is only used in the plain PyTorch+Geometric version
from torch.utils.tensorboard import SummaryWriter
# these imports are only used in the Lighning version
import pytorch_lightning as pl
import torch.nn.functional as F


class GCN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.num_features = kwargs["num_features"] \
                    if "num_features" in kwargs.keys() else 3
        self.num_classes = kwargs["num_classes"] \
                    if "num_classes" in kwargs.keys() else 2
        # hidden layer node features
        self.hidden = 256
        self.model = Sequential("x, edge_index, batch_index", [
                (GCNConv(self.num_features, self.hidden),
                    "x, edge_index -> x1"),
                (ReLU(), "x1 -> x1a"),
                (Dropout(p=0.5), "x1a -> x1d"),
                (GCNConv(self.hidden, self.hidden), "x1d, edge_index -> x2"),
                (ReLU(), "x2 -> x2a"),
                (Dropout(p=0.5), "x2a -> x2d"),
                (GCNConv(self.hidden, self.hidden), "x2d, edge_index -> x3"),
                (ReLU(), "x3 -> x3a"),
                (Dropout(p=0.5), "x3a -> x3d"),
                (GCNConv(self.hidden, self.hidden), "x3d, edge_index -> x4"),
                (ReLU(), "x4 -> x4a"),
                (Dropout(p=0.5), "x4a -> x4d"),
                (GCNConv(self.hidden, self.hidden), "x4d, edge_index -> x5"),
                (ReLU(), "x5 -> x5a"),
                (Dropout(p=0.5), "x5a -> x5d"),
                (global_mean_pool, "x5d, batch_index -> x6"),
                (Linear(self.hidden, self.num_classes), "x6 -> x_out")])

    def forward(self, x, edge_index, batch_index):
        x_out = self.model(x, edge_index, batch_index)
        return x_out

    def training_step(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch
        x_out = self.forward(x, edge_index, batch_index)
        loss = F.cross_entropy(x_out, batch.y)
        # metrics here
        pred = x_out.argmax(-1)
        label = batch.y
        accuracy = (pred == label).sum() / pred.shape[0]

        self.log("loss/train", loss)
        self.log("accuracy/train", accuracy)

        return loss

    def validation_step(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch
        x_out = self.forward(x, edge_index, batch_index)
        loss = F.cross_entropy(x_out, batch.y)
        pred = x_out.argmax(-1)
        return x_out, pred, batch.y

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in validation_step_outputs:
            val_loss += F.cross_entropy(output, labels, reduction="sum")
            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]
            val_accuracy = num_correct / num_total
            val_loss = val_loss / num_total

        self.log("accuracy/val", val_accuracy)
        self.log("loss/val", val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 3e-4)
