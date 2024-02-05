"""
Created by Fabio Hellmann.
"""


import os
import pathlib
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

# i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise).
from lib.utils import glob_dir

SEED = 42

label_map = {
    # Neutral
    0: 0,
    # Angry
    1: 6,
    # Contempt
    2: 7,
    # Disgust
    3: 5,
    # Fear
    4: 4,
    # Happy
    5: 1,
    # Sad
    6: 2,
    # Surprise
    7: 3
}


class LabelExtractor(ABC):
    """
    An extractor for the labels of the dataset.
    """

    def __init__(self, dataset_dir: str):
        """
        Initialize the LabelExtractor.
        @param dataset_dir: The directory to the dataset.
        """
        self.dataset_dir = dataset_dir

    @abstractmethod
    def __call__(self) -> pd.DataFrame:
        """
        Extract the labels from the dataset.
        @return: The labels as pandas.DataFrame.
        """
        raise NotImplementedError()


class CelebALabelExtractor(LabelExtractor):
    def __init__(self, dataset_dir: str):
        super().__init__(os.path.join(dataset_dir, 'celeba'))

    def __call__(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.dataset_dir, 'list_eval_partition.txt'), delimiter=' ',
                         header=None)
        df = df.rename(columns={0: 'image_path', 1: 'split'})
        df = df.merge(
            pd.read_csv(os.path.join(self.dataset_dir, 'list_attr_celeba.txt'), delimiter=r'\s+',
                        skiprows=1).rename_axis('image_path').reset_index(), on='image_path',
            how='left')
        labels = list(df.keys())[2:]
        df[labels] = (df[labels] + 1) / 2
        return df


class CKPlusLabelExtractor(LabelExtractor):
    def __init__(self, dataset_dir: str):
        super().__init__(dataset_dir)

    def __call__(self) -> pd.DataFrame:
        label_base_dir = os.path.join(self.dataset_dir, 'Emotion')
        image_base_dir = os.path.join(self.dataset_dir, 'cohn-kanade-images')
        val_split = 0.2
        result = {
            'subject_id': [],
            'image_path': [],
            'anger': [],
            'contempt': [],
            'disgust': [],
            'fear': [],
            'happy': [],
            'sadness': [],
            'surprise': [],
        }
        labels = list(result.keys())[2:]
        for file in tqdm(glob_dir(image_base_dir)):
            folder_structure = file.replace(image_base_dir, '').split(os.sep)
            img_path = os.path.join(*folder_structure[:-1], os.path.basename(file))
            labels_path = os.path.join(label_base_dir, *folder_structure[:-1])
            label_path = os.path.join(labels_path, f'{pathlib.Path(file).stem}_emotion.txt')
            # creating on hot encoding
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='UTF-8') as label_file:
                    mapped_idx = int(float(label_file.readline().strip()))
                    sid = folder_structure[-3]
                    result['subject_id'].append(sid)
                    result['image_path'].append(img_path)
                    for idx, key in enumerate(labels):
                        result[key].append(1 if idx == mapped_idx - 1 else 0)
        df = pd.DataFrame(result)
        df_grouped = df.groupby(['subject_id'])[labels].sum().sample(frac=val_split,
                                                                     random_state=SEED)
        df_train = df.loc[~df['subject_id'].isin(df_grouped.index)]
        df_train['split'] = [0] * len(df_train)
        df_val = df.loc[df['subject_id'].isin(df_grouped.index)]
        df_val['split'] = [1] * len(df_val)
        return pd.concat([df_train, df_val]).drop('subject_id', axis=1)


class AffectNetLabelExtractor(LabelExtractor):
    def __init__(self, dataset_dir: str):
        super().__init__(dataset_dir)

    def __call__(self) -> pd.DataFrame:
        label_base_dir = os.path.join(self.dataset_dir, 'Manually_Annotated_file_lists')
        result = self.parse_csv_images(os.path.join(label_base_dir, 'training.csv'), 0)
        result.update(self.parse_csv_images(os.path.join(label_base_dir, 'validation.csv'), 1))
        df = pd.DataFrame(result)
        return df

    def parse_csv_images(self, csv_file: str, split: int):
        result = {
            'image_path': [],
            'neutral': [],
            'anger': [],
            'contempt': [],
            'disgust': [],
            'fear': [],
            'happy': [],
            'sadness': [],
            'surprise': [],
            'split': [],
        }
        df = pd.read_csv(csv_file)
        for index, row in tqdm(df.iterrows()):
            label = row['expression']
            if label > 7:
                continue
            result['image_path'].append(row['#subDirectory_filePath'])
            for idx, key in enumerate(list(result.keys())[2:-1]):
                result[key].append(1 if idx == label else 0)
            result['split'].append(split)
        return result


class FacesLabelExtractor(LabelExtractor):
    def __init__(self, dataset_dir: str):
        super().__init__(dataset_dir)

    def __call__(self) -> pd.DataFrame:
        emotion_map = {
            'n': 0,  # Neutrality
            'h': 1,  # Happy
            's': 2,  # Sadness
            'f': 3,  # Fear
            'd': 4,  # Disgust
            'a': 5,  # Anger
        }
        result = {
            'image_path': [],
            'subject_id': [],
            'neutral': [],
            'happy': [],
            'sadness': [],
            'fear': [],
            'disgust': [],
            'anger': [],
        }
        val_set = ['004', '066', '079', '116', '140',
                   '168']  # Subject photos officially usable for publications
        val_split = 0.2
        labels = list(result.keys())[2:]
        for file in tqdm(glob_dir(os.path.join(self.dataset_dir, 'bilder'))):
            file_split = pathlib.Path(file).stem.split('_')
            subject_id = file_split[0]
            emotion = file_split[3]
            result['image_path'].append(os.path.basename(file))
            result['subject_id'].append(subject_id)
            mapped_idx = emotion_map[emotion]
            for idx, key in enumerate(labels):
                result[key].append(1 if idx == mapped_idx else 0)
        df = pd.DataFrame(result)
        df_grouped = df.drop(val_set).groupby(['subject_id'])[list(result.keys())[2:]].sum().sample(
            frac=val_split)
        df_train = df[~df['subject_id'].isin(df_grouped.index)]
        df_train['split'] = [0] * len(df_train)
        df_val = df[df['subject_id'].isin(df_grouped.index)]
        df_val['split'] = [1] * len(df_val)
        return pd.concat([df_train, df_val]).drop('subject_id', axis=1)


label_extractors = {
    'CelebA': CelebALabelExtractor,
    'CK+': CKPlusLabelExtractor,
    'AffectNet': AffectNetLabelExtractor,
    'FACES': FacesLabelExtractor,
}


def extract_labels(dataset_path: str) -> Optional[str]:
    """
    Extract the labels from the given dataset and bring it in a uniform format.
    @param dataset_path: The path to the dataset.
    @return: The path to the csv file or None if no extractor could be found.
    """
    path_parts = dataset_path.split(os.sep)
    for dataset_name in path_parts:
        if dataset_name in label_extractors:
            dataset_path = dataset_path[:dataset_path.index(dataset_name) + len(dataset_name)]
            logger.debug(f'Extracting labels for {dataset_name}...')
            # csv_file =
            extractor_class = label_extractors[dataset_name]
            extractor = extractor_class(dataset_path)
            df = extractor()
            csv_file = os.path.join(dataset_path, 'labels.csv')
            df.to_csv(csv_file)
            logger.info(f'Extracted labels from {dataset_name} to {dataset_path}')
            df_anaylser = df.drop(['image_path'], axis=1)
            meta_data = df_anaylser.drop(["split"], axis=1).sum()
            logger.debug(f'Meta-Data:\n{meta_data}')
            train_split = df_anaylser[df_anaylser["split"] == 0].drop(["split"], axis=1).sum()
            logger.debug(f'Train-Split:\n{train_split}')
            val_split = df_anaylser[df_anaylser["split"] == 1].drop(["split"], axis=1).sum()
            logger.debug(f'Validation-Split:\n{val_split}')
            return csv_file
    logger.warning(f'No label extractor found for {dataset_path}')
    return None
