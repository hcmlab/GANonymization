"""
Created by Fabio Hellmann.
"""

from typing import List

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.cli import ReduceLROnPlateau
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from lib.datasets.labeled_dataset import LabeledDataset


class GenericClassifier(pl.LightningModule):
    """
    The GenericClassifier is based on the ConvNeXt architecture and
    can be used to train a multi-class or multi-label problem.
    """

    weights = ConvNeXt_Base_Weights.DEFAULT

    def __init__(self, multi_label: bool, classes: List[str], batch_size: int,
                 learning_rate: float, num_workers: int, device: str, val_db: LabeledDataset = None,
                 train_db: LabeledDataset = None, class_weights: Tensor = None):
        """
        Create a new GenericClassifier.
        @param multi_label: If the problem at hand is multi-label then True otherwise False for
        multi-class.
        @param classes: The class names as a list.
        @param batch_size: The size of the batches.
        @param learning_rate: The learning rate.
        @param num_workers: The number of cpu-workers.
        @param device: The device to run the model on.
        @param val_db: The validation dataset.
        @param train_db: The training dataset.
        @param class_weights: The class weights of the dataset.
        """
        super().__init__()
        # Settings
        self.train_db = train_db
        self.val_db = val_db
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multi_label = multi_label
        self.classes = classes
        self.num_classes = len(classes)
        # Define model architecture
        self.model = convnext_base(weights=self.weights)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features,
                                              self.num_classes)
        self.class_weights = class_weights.float().to(device)
        # Loss function, accuracy
        if self.multi_label:
            logger.debug('Model is a Multi-Label Classificator')
            self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
            self.activation = torch.nn.Sigmoid().to(device)
            self.metrics = {
                'accuracy': MultilabelAccuracy(num_labels=self.num_classes).to(device)
            }
        else:
            logger.debug('Model is a Multi-Class Classificator')
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights).to(device)
            self.activation = torch.nn.Softmax(dim=1).to(device)
            self.metrics = {
                'top_1_acc': MulticlassAccuracy(top_k=1, num_classes=self.num_classes).to(device),
                'top_5_acc': MulticlassAccuracy(top_k=5, num_classes=self.num_classes).to(device),
            }
        self.save_hyperparameters(ignore=['val_db', 'train_db'])

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.activation(self.forward(x))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.learning_rate,
                                      weight_decay=0.001)
        scheduler = ReduceLROnPlateau(optimizer, monitor='val_loss', patience=3)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}]

    def training_step(self, train_batch, batch_idx):
        loss, _ = self.step('train', train_batch)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_db, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def validation_step(self, val_batch, batch_idx):
        _, labels = val_batch
        _, output = self.step('val', val_batch)
        if not self.multi_label:
            labels = torch.argmax(labels, dim=1)
        for key, metric in self.metrics.items():
            self.log(f'val_{key}', metric(output, labels), prog_bar=True)

    def val_dataloader(self):
        return DataLoader(self.val_db, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def step(self, tag: str, batch):
        images, labels = batch
        output = self.forward(images)
        if self.multi_label:
            loss = self.loss_function(output, labels)
            loss = (loss * self.class_weights).mean()
        else:
            loss = self.loss_function(output, torch.argmax(labels, dim=1))
        self.log(f'{tag}_loss', loss.item(), prog_bar=True)
        return loss, output
