"""
Created by Fabio Hellmann.
"""

import os.path

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from loguru import logger
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix, MultilabelConfusionMatrix
from torchvision.transforms import Compose
from tqdm import tqdm

from lib.datasets import DatasetSplit
from lib.datasets.labeled_dataset import LabeledDataset
from lib.models.generic_classifier import GenericClassifier
from lib.utils import get_last_ckpt


def eval_classifier(models_dir: str, data_dir: str, batch_size: int = 128, device: int = 0,
                    output_dir: str = 'output', num_workers: int = 2):
    logger.info(f"Parameters: {', '.join([f'{key}: {value}' for key, value in locals().items()])}")
    os.makedirs(output_dir, exist_ok=True)
    val_db = LabeledDataset(data_dir, DatasetSplit.VALIDATION,
                            Compose([GenericClassifier.weights.transforms()]))
    model = GenericClassifier.load_from_checkpoint(checkpoint_path=get_last_ckpt(models_dir),
                                                   classes=val_db.classes,
                                                   multi_label=val_db.is_multi_label)
    model.eval()
    model.to(device)
    if model.multi_label:
        cm = MultilabelConfusionMatrix(num_labels=val_db.num_classes, normalize='true').to(device)
        logger.debug('Setting up evaluation for multi-label classification')
    else:
        cm = MulticlassConfusionMatrix(num_classes=val_db.num_classes, normalize='true').to(device)
        logger.debug('Setting up evaluation for multi-class classification')
    dl = DataLoader(val_db, batch_size=batch_size, num_workers=num_workers)
    all_label = []
    all_pred = []
    for batch in tqdm(dl, desc='Evaluation'):
        images, labels = batch
        pred = model.predict(images.to(device))
        labels = labels.to(device)
        if val_db.is_multi_label:
            ml_pred = torch.round(pred).int()
            ml_labels = labels.int()
            cm.update(ml_pred, ml_labels)
        else:
            mc_labels = torch.argmax(labels, dim=1)
            cm.update(pred, mc_labels)
        all_label.extend(labels.cpu().detach().numpy())
        all_pred.extend(pred.cpu().detach().numpy())
    # Confusion Matrix
    all_label = np.asarray(all_label)
    all_pred = np.asarray(all_pred)
    np.savez_compressed(os.path.join(output_dir, 'result_pred_label.npz'), pred=all_pred,
                        label=all_label)
    conf_matrix = cm.compute()
    conf_matrix = conf_matrix.cpu().detach().numpy()
    logger.debug('Creating Confusion Matrix...')
    if model.multi_label:
        n_columns = int(model.num_classes / 4)
        n_rows = int(model.num_classes / n_columns)
        fig = plt.figure(figsize=(n_columns * 2, n_rows * 2))
        plt.tight_layout()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(n_rows, n_columns),
                         axes_pad=0.5,
                         share_all=True,
                         cbar_mode='single',
                         cbar_location='right',
                         cbar_size='5%',
                         cbar_pad=0.5)
        for idx in tqdm(range(conf_matrix.shape[0]), desc='Multi-Label Confusion Matrices'):
            label = model.classes[idx]
            binary_labels = ['Present', 'Absent']
            df_cm = pd.DataFrame(conf_matrix[idx, :, :], index=binary_labels, columns=binary_labels)
            df_cm.to_csv(os.path.join(output_dir,
                                      f'{os.path.basename(data_dir)}_{label}_confusion_matrix.csv'))
            sn.set(font_scale=1.3)
            sn.heatmap(df_cm, annot=True, fmt='.2f', xticklabels=binary_labels,
                       yticklabels=binary_labels,
                       ax=grid[idx], cbar_ax=grid[0].cax, annot_kws={'fontsize': 13})
            grid[idx].set_ylabel('Actual')
            grid[idx].set_xlabel('Predicted')
            grid[idx].set_title(label.replace('_', ' ').replace('-', ' '))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{os.path.basename(data_dir)}_confusion_matrix.png'))
        plt.close()
    else:
        df_cm = pd.DataFrame(conf_matrix, index=val_db.labels, columns=val_db.labels)
        df_cm.to_csv(os.path.join(output_dir, f'{os.path.basename(data_dir)}_confusion_matrix.csv'))
        plt.figure(figsize=(
            len(val_db.labels) + int(len(val_db.labels) / 2),
            len(val_db.labels) + int(len(val_db.labels) / 2)))
        plt.tight_layout()
        sn.set(font_scale=1.3)
        sn.heatmap(df_cm, annot=True, fmt='.2f', xticklabels=val_db.labels,
                   yticklabels=val_db.labels, annot_kws={
                'fontsize': 13
            })
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(output_dir, f'{os.path.basename(data_dir)}_confusion_matrix.png'))
        plt.close()
    logger.debug('Confusion Matrix created!')
    # Classification Report
    logger.debug('Creating Classification Report...')
    if model.multi_label:
        all_label = np.round(all_label)
        all_pred = np.round(all_pred)
    else:
        all_label = np.squeeze(np.argmax(all_label, axis=1))
        all_pred = np.squeeze(np.argmax(all_pred, axis=1))
    report = classification_report(all_label, all_pred, target_names=val_db.labels)
    with open(os.path.join(output_dir, f'{os.path.basename(data_dir)}_classification_report.txt'),
              'w+', encoding='UTF-8') as f:
        f.write(report)
    logger.debug('Classification Report created!')
