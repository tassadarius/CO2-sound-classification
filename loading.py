"""
CO2 - Wintersemester 2019-2020
Sound classification with neural networks
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
from typing import List, Tuple
import tensorflow as tf

Audio = namedtuple('Audio', ['data', 'name', 'sample_rate'])


def load_audio(audio_path: str, audio_format: str, filter_list: str = None) -> List[namedtuple]:
    """ Load audio data and convert it to numpy array

    :param audio_path:
    :param audio_format:
    :return:
    """
    _path = Path(audio_path)
    if not _path.is_dir():
        raise ValueError("audio_path must be an existing directory")

    # handle exception to check if there isn't a file which is corrupted
    data = []
    CTR = 0
    for file in _path.rglob(f"*.{audio_format}"):
        if (filter_list and file.stem in filter_list) or not filter_list:
            # here kaiser_fast is a technique used for faster extraction
            audio_data, sample_rate = librosa.load(file, res_type='kaiser_fast')
            # fft_data = np.fft.fft(audio_data)
            fft_data = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            tmp_audio = Audio(data=fft_data, name=file.stem, sample_rate=sample_rate)
            data.append(tmp_audio)
            CTR += 1
            if CTR > 10:
                break

    return data


def load_labels(label_path: str, relevant_columns: Tuple[str, str], sep=','):
    """ Load the labels from a csv file

    :param label_path:
    :param sep:
    :return:
    """
    label_mapping = {}
    df = pd.read_csv(label_path, sep=sep)
    df.apply(lambda x: label_mapping.update({str(x[relevant_columns[0]]):  x[relevant_columns[1]]}), axis=1)
    return label_mapping


def labels_one_hot(labels: List[str]):
    # Convert the strings to pandas categories. It's easy to get strings to numbers with .codes
    categories = pd.Categorical(labels)
    labels_numerical = categories.codes
    # We need the amount of categories
    category_count = len(categories.categories)
    labels_numpy = np.array(labels_numerical).reshape(-1)
    labels_encoded = np.eye(category_count)[labels_numpy]
    return tf.constant(labels_encoded, dtype=tf.int8)


class DataProvider:

    def __init__(self, data: Audio, labels: List[str]):
        self.labels_encoded = labels_one_hot(labels)
        self.only_data = [np.ndarray.flatten(x.data) for x in data]
        assert (len(self.labels_encoded) == len(self.only_data))


    def __call__(self):
        for data, label in zip(self.only_data, self.labels_encoded):
            yield tf.constant(data), label


def prepare_data(wav_directory: str, label_csv: str):
    audio_labels = load_labels(label_csv, ('ID', 'Class'))
    audio_data = load_audio(wav_directory, 'wav', filter_list=audio_labels.keys())

    return DataProvider(audio_data, [audio_labels[x.name] for x in audio_data])
