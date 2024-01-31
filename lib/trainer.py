"""
Created by Fabio Hellmann.
"""

import os
import random

import numpy as np
import pytorch_lightning
import torch
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def setup_torch_device(device: int, seed: int) -> str:
    """
    Set up the torch device with a seed.
    @param device: The device the model should be run on.
    @param seed: The seed for the run.
    @return: The string of the device.
    """
    torch_device = f'cuda:{device}' if device >= 0 else 'cpu'
    logger.info(f"Using {'GPU' if device >= 0 else 'CPU'}.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    pytorch_lightning.seed_everything(seed)
    if device >= 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    return torch_device


def setup(model: LightningModule, log_dir: str, models_dir: str, num_epoch: int, device: int,
          save_top_k: int = 0, monitor: str = None, metric_mode: str = None, early_stop_n: int = 0,
          checkpoint_interval: int = None) -> pytorch_lightning.Trainer:
    """
    Run the lightning model.
    @param model: The model to train.
    @param log_dir: The directory to save the logs.
    @param models_dir: The directory to save the models.
    @param num_epoch: The number of epochs to train.
    @param device: The device to work on.
    @param save_top_k: Save top k checkpoints - or every epoch if k=0.
    @param monitor: The metric variable to monitor.
    @param metric_mode: The mode of the metric to decide which checkpoint to choose (min or max).
    @param early_stop_n: Stops training after n epochs of no improvement - default is deactivated.
    @param checkpoint_interval: The interval a checkpoint should be saved if save_top_k is 0.
    """
    model_name = model.__class__.__name__
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(log_dir, name='', version='')
    callbacks = [LearningRateMonitor()]
    if save_top_k > 0:
        callbacks.append(
            ModelCheckpoint(dirpath=models_dir, filename='{epoch}-{' + monitor + ':.6f}',
                            save_top_k=save_top_k, monitor=monitor, mode=metric_mode))
    else:
        callbacks.append(
            ModelCheckpoint(dirpath=models_dir, filename='{epoch}-{step}', monitor='step',
                            mode='max',
                            save_top_k=-1, every_n_train_steps=checkpoint_interval))
    if early_stop_n > 0:
        callbacks.append(
            EarlyStopping(monitor=monitor, min_delta=0.00, patience=early_stop_n, verbose=False,
                          mode=metric_mode))

    trainer = pytorch_lightning.Trainer(deterministic=True,
                                        accelerator="gpu" if device >= 0 else "cpu",
                                        devices=[device] if device >= 0 else None,
                                        callbacks=callbacks,
                                        logger=tb_logger,
                                        max_epochs=num_epoch,
                                        val_check_interval=checkpoint_interval,
                                        log_every_n_steps=1,
                                        detect_anomaly=True)
    logger.info(
        f'Train {model_name} for {num_epoch} epochs and save logs under {tb_logger.log_dir} '
        f'and models under {models_dir}')
    model.train()
    model.to(device)
    return trainer
