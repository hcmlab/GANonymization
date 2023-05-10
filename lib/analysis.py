import math
import os.path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from deepface import DeepFace
from loguru import logger
from matplotlib import pyplot as plt
from scipy.stats import shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from lib.datasets import DatasetSplit
from lib.datasets.labeled_dataset import LabeledDataset
from lib.models.generic_classifier import GenericClassifier
from lib.transform import FaceSegmentation, Pix2PixTransformer


def emotion_analysis(db_path: str, result_path: str):
    db_name = os.path.basename(db_path)
    logger.debug(f'Analysis for {db_name}')

    # Load labeled datasets
    db_seg = LabeledDataset(os.path.join(db_path, 'segmentation'), DatasetSplit.VALIDATION, Compose([]))
    db_gan = LabeledDataset(os.path.join(db_path, 'ganonymization'), DatasetSplit.VALIDATION, Compose([]))
    db_dp2 = LabeledDataset(os.path.join(db_path, 'deepprivacy2'), DatasetSplit.VALIDATION, Compose([]))

    # Load predicted labels
    npz_seg = np.load(os.path.join(result_path, 'segmentation', 'result_pred_label.npz'))
    npz_gan = np.load(os.path.join(result_path, 'ganonymization', 'result_pred_label.npz'))
    npz_dp2 = np.load(os.path.join(result_path, 'deepprivacy2', 'result_pred_label.npz'))

    # Create dataframes with predicted labels and image names
    df_seg = db_seg.meta_data
    df_gan = db_gan.meta_data
    df_dp2 = db_dp2.meta_data

    df_seg_pred = pd.DataFrame(npz_seg['pred'], columns=db_seg.classes)
    df_gan_pred = pd.DataFrame(npz_gan['pred'], columns=db_seg.classes)
    df_dp2_pred = pd.DataFrame(npz_dp2['pred'], columns=db_seg.classes)

    df_seg_pred['image_name'] = [os.path.basename(f).split('-')[-1] for f in df_seg['image_path']]
    df_gan_pred['image_name'] = [os.path.basename(f).split('-')[-1] for f in df_gan['image_path']]
    df_dp2_pred['image_name'] = [os.path.basename(f).split('-')[-1] for f in df_dp2['image_path']]

    # Calculate mean distance between original and synthesized images for each category
    synthesized_label = [f'{c}_x' for c in db_seg.classes]
    original_label = [f'{c}_y' for c in db_seg.classes]

    df = pd.DataFrame(index=[c.replace('_', ' ') for c in db_seg.classes])

    df_seg_gan_pred = pd.merge(df_gan_pred, df_seg_pred, on='image_name')
    df_seg_dp2_pred = pd.merge(df_dp2_pred, df_seg_pred, on='image_name')

    df['GANonymization'] = [(df_seg_gan_pred[x] - df_seg_gan_pred[y]).abs().mean() for x, y in zip(synthesized_label, original_label)]
    df['DeepPrivacy2'] = [(df_seg_dp2_pred[x] - df_seg_dp2_pred[y]).abs().mean() for x, y in zip(synthesized_label, original_label)]

    # Save results
    df.to_csv(os.path.join(result_path, f'{db_name}_mean_distance.csv'))
    df.to_latex(os.path.join(result_path, f'{db_name}_mean_distance.latex'))
    if len(original_label) < 20:
        fig_width = int(round(len(df.index) / 2))
        if fig_width < 4:
            fig_width = 4
        df.plot.bar(figsize=(fig_width, 4))
        plt.ylabel('Mean Distance')
        plt.xlabel('Category')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f'{db_name}_mean_distance.png'))
        plt.close()
    else:
        df_split = np.array_split(df, int(round(len(original_label) / 20)))
        for idx, df_tmp in enumerate(df_split):
            fig_width = int(round(len(df_tmp.index) / 2))
            if fig_width < 4:
                fig_width = 4
            df_tmp.plot.bar(figsize=(fig_width, 4))
            plt.ylabel('Mean Distance')
            plt.xlabel('Category')
            plt.legend(loc='upper center')
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f'{db_name}_{idx}_mean_distance.png'))
            plt.close()

    # T-Test
    df_seg_gan_pred = pd.merge(df_gan_pred, df_seg_pred, left_on='image_name', right_on='image_name')
    ganonymization = np.asarray(
        [(df_seg_gan_pred[x] - df_seg_gan_pred[y]).abs().to_numpy() for x, y in zip(synthesized_label, original_label)])

    df_seg_dp2_pred = pd.merge(df_dp2_pred, df_seg_pred, left_on='image_name', right_on='image_name')[
        df_dp2_pred['image_name'].isin(df_gan_pred['image_name'])]
    deepprivacy2 = np.asarray(
        [(df_seg_dp2_pred[x] - df_seg_dp2_pred[y]).abs().to_numpy() for x, y in zip(synthesized_label, original_label)])

    ganonymization = ganonymization.transpose()
    deepprivacy2 = deepprivacy2.transpose()

    results = {}
    for col_idx in range(len(ganonymization[0])):
        _, pvalue = shapiro(np.concatenate([ganonymization[:, col_idx], deepprivacy2[:, col_idx]]))
        logger.debug(f'Shapiro: {pvalue}')
        result = wilcoxon(ganonymization[:, col_idx], deepprivacy2[:, col_idx], method='approx')
        n = len(ganonymization[:, col_idx])
        r = result.zstatistic / math.sqrt(n)
        logger.info(
            f'{wilcoxon.__name__}-{original_label[col_idx]}: p-value={result.pvalue}, statistic={result.statistic}, zstatistic={result.zstatistic}, n={n}, r={r}')
        results.setdefault('pvalue', []).append(result.pvalue)
        # results.setdefault('statistic', []).append(result.statistic)
        results.setdefault('zstatistic', []).append(result.zstatistic)
        results.setdefault('r', []).append(r)
        results.setdefault('n', []).append(n)

    reject, pvals_corrected, _, _ = multipletests(np.asarray(results['pvalue']), method='bonferroni')
    logger.info(f'\nP-Values (corrected): {pvals_corrected}')

    df = pd.DataFrame(results,
                      index=[c.replace('_', ' ') for c in db_seg.classes])
    df.rename({'pvalue': 'P-Value', 'zstatistic': 'Z-Statistic', 'n': 'N'}, axis=1, inplace=True)
    df.to_csv(os.path.join(result_path, f'statistic_pvalue.csv'))
    df.to_latex(os.path.join(result_path, f'statistic_pvalue.latex'))


def face_comparison_analysis(output_dir: str, img_orig_path: str, *img_paths, img_size: int = 512):
    def original_filename(file):
        return file.split('-')[-1]

    os.makedirs(output_dir, exist_ok=True)
    sub_dir = 'val'
    img_dirs = {
        os.path.basename(path): [os.path.join(path, sub_dir, f) for f in os.listdir(os.path.join(path, sub_dir))]
        for path in img_paths}
    img_dirs_short = {method: [original_filename(path) for path in paths] for method, paths in img_dirs.items()}

    comp_result = []
    transformers = Compose([Resize(img_size), ToTensor()])
    for orig_img in tqdm(os.listdir(os.path.join(img_orig_path, sub_dir)), desc='Comparison'):
        images = {'original': os.path.join(img_orig_path, sub_dir, orig_img)}
        for method, paths in img_dirs.items():
            if original_filename(orig_img) in img_dirs_short[method]:
                images[method] = img_dirs[method][
                    [original_filename(p) for p in paths].index(original_filename(orig_img))]
        if len(images) < len(img_paths) + 1:
            continue
        for idx, items in enumerate(images.items()):
            db_name, img_path = items
            # Predict similarity of faces
            verify_result = {'distance': 0, 'threshold': 0}
            if idx > 0:
                tmp_result = DeepFace.verify(images['original'], img_path, model_name="VGG-Face",
                                             detector_backend='skip', enforce_detection=False, align=False)
                if len(verify_result) > 0:
                    verify_result = tmp_result
                comp_result.append([db_name, original_filename(img_path), verify_result['distance'],
                                    verify_result['threshold']])
        img_sample = torch.cat(
            [transformers(Image.open(images['original'])), transformers(Image.open(images['deepprivacy2'])),
             transformers(Image.open(images['ganonymization']))], -2)
        save_image(img_sample, os.path.join(output_dir, original_filename(orig_img)), normalize=True)
    df = pd.DataFrame(comp_result, columns=['dataset', 'filename', 'distance', 'threshold'])
    df.to_csv(os.path.join(output_dir, 'prediction_result.csv'))
    df_distance_mean = df.groupby(['dataset'])['distance'].mean()
    threshold_mean = df.groupby(['dataset'])['threshold'].mean()['ganonymization']
    df_distance_mean.to_csv(os.path.join(output_dir, 'mean_prediction_result.csv'))
    df_distance_mean.transpose().plot.bar(rot=0)
    plt.axhline(y=threshold_mean, ls='dashed', color='b')
    plt.ylabel('Mean Cosine Distance')
    plt.xlabel('Anonymization Method')
    plt.savefig(os.path.join(output_dir, 'mean_distance_per_method.png'))
    plt.close()

    logger.debug(f'Analyze comparison for: {output_dir}')
    df = pd.read_csv(os.path.join(output_dir, 'prediction_result.csv'))
    df_gan = df[df['dataset'] == 'ganonymization'].reset_index()
    df_dp2 = df[df['dataset'] == 'deepprivacy2'].reset_index()
    df_gan_nan = df_gan[df_gan['distance'].isna()]['filename'].index
    df_gan.drop(df_gan_nan, inplace=True)
    df_dp2.drop(df_gan_nan, inplace=True)
    df_gan = df_gan['distance'].to_numpy()
    df_dp2 = df_dp2['distance'].to_numpy()
    _, pvalue = shapiro(np.concatenate([df_gan, df_dp2]))
    logger.debug(f'Shapiro: {pvalue}')
    result = wilcoxon(df_gan, df_dp2, method='approx')
    n = len(df_gan)
    r = result.zstatistic / math.sqrt(n)
    logger.info(
        f'{wilcoxon.__name__}: p-value={result.pvalue}, statistic={result.statistic}, zstatistic={result.zstatistic}, n={n}, r={r}')
    reject, pvals_corrected, _, _ = multipletests(result.pvalue, method='bonferroni')
    logger.info(
        f'{wilcoxon.__name__}: p-value={result.pvalue}, statistic={result.statistic}, zstatistic={result.zstatistic}, n={n}, r={r}')

    df = pd.DataFrame({'P-Value': pvals_corrected, 'Z-Statistic': result.zstatistic, 'r': r, 'N': n})
    df.to_csv(os.path.join(output_dir, f'statistic_pvalue.csv'))
    df.to_latex(os.path.join(output_dir, f'statistic_pvalue.latex'))


def facial_traits_analysis(data_dir: str, model_file: str, output_dir: str):
    db_seg = LabeledDataset(os.path.join(data_dir, FaceSegmentation.__class__.__name__), DatasetSplit.VALIDATION,
                            Compose([GenericClassifier.weights.transforms()]))
    db_seg.meta_data['image_name'] = db_seg.meta_data['image_path'].apply(
        lambda f: os.path.basename(f).split('-')[-1]).values.tolist()
    db_gan = LabeledDataset(os.path.join(data_dir, Pix2PixTransformer.__class__.__name__), DatasetSplit.VALIDATION,
                            Compose([GenericClassifier.weights.transforms()]))
    db_gan.meta_data['image_name'] = db_gan.meta_data['image_path'].apply(
        lambda f: os.path.basename(f).split('-')[-1]).values.tolist()
    db_seg.meta_data.drop(db_seg.meta_data[~db_seg.meta_data['image_name'].isin(db_gan.meta_data['image_name'])].index,
                          inplace=True)
    db_seg.meta_data.reset_index(inplace=True)
    model = GenericClassifier.load_from_checkpoint(model_file, classes=db_seg.classes)
    result_positive = {label: 0 for label in db_seg.classes}
    result_negative = {label: 0 for label in db_seg.classes}
    for idx in tqdm(range(len(db_seg)), total=len(db_seg)):
        pred_seg = model.predict(torch.unsqueeze(db_seg[idx][0].to(model.device), dim=0))[0]
        if torch.any(pred_seg > 0.5):
            pred_gan = model.predict(torch.unsqueeze(db_gan[idx][0].to(model.device), dim=0))[0]
            pred_seg = pred_seg.cpu().detach().numpy()
            pred_gan = pred_gan.cpu().detach().numpy()
            for label_idx in range(db_seg.num_classes):
                if pred_seg[label_idx] > 0.5 and pred_gan[label_idx] > 0.5:
                    result_positive[db_seg.classes[label_idx]] += 1
                elif pred_seg[label_idx] > 0.5 and pred_gan[label_idx] <= 0.5:
                    result_negative[db_seg.classes[label_idx]] += 1
    df_pos = pd.read_csv(os.path.join(output_dir, 'result', 'positive.csv'), index_col=0)
    df_neg = pd.read_csv(os.path.join(output_dir, 'result', 'negative.csv'), index_col=0)
    df = pd.merge(df_pos, df_neg, right_index=True, left_index=True)
    result = df['Negative'] / df.sum(axis=1)
    result = pd.DataFrame(result, columns=['GANonymization'])
    result = result.rename(index={s: s.replace('_', ' ') for s in result.index.values})
    result = result.sort_values('GANonymization', ascending=False)
    result.to_csv(os.path.join(output_dir, 'result', 'traits_removed.csv'))
    result.to_latex(os.path.join(output_dir, 'result', 'traits_removed.latex'))
    df_split = np.array_split(result, int(round(len(result) / 20)))
    for idx, df_tmp in enumerate(df_split):
        fig_width = int(round(len(df_tmp.index) / 2))
        if fig_width < 4:
            fig_width = 4
        df_tmp.plot.bar(figsize=(fig_width, 4))
        plt.gca().set_ylim([0, 1])
        plt.ylabel('Removed (in %)')
        plt.xlabel('Category')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'result', f'{idx}_traits_removed.png'))
        plt.close()
