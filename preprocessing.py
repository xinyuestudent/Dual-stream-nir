"""Data preprocessing utilities for spectral, audio, ECG, and EEG inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, savgol_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ArrayDataset(Dataset):
    """Simple dataset for numpy arrays or tensors."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    class_names: Optional[Sequence[str]] = None


def baseline_correction(input_data: np.ndarray) -> np.ndarray:
    """Correct spectral baseline using minima and a quadratic spline."""

    corrected = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        peaks, _ = find_peaks(-input_data[i])
        if len(peaks) < 3:
            corrected[i] = input_data[i] - np.min(input_data[i])
            continue
        spline = UnivariateSpline(peaks, input_data[i][peaks], s=0, k=2)
        baseline = spline(range(len(input_data[i])))
        corrected[i] = input_data[i] - baseline
    return corrected


def snv(input_data: np.ndarray) -> np.ndarray:
    """Standard normal variate normalization per sample."""

    mean = np.mean(input_data, axis=1, keepdims=True)
    std = np.std(input_data, axis=1, keepdims=True)
    return (input_data - mean) / (std + 1e-8)


def msc(input_data: np.ndarray) -> np.ndarray:
    """Multiplicative scatter correction."""

    mean_spectrum = np.mean(input_data, axis=0)
    corrected = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        coef = np.polyfit(mean_spectrum, input_data[i], 1)
        corrected[i] = (input_data[i] - coef[1]) / (coef[0] + 1e-8)
    return corrected


def first_derivative(input_data: np.ndarray, window_size: int = 5, poly_order: int = 2) -> np.ndarray:
    """Savitzky-Golay first derivative."""

    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order, deriv=1)


def add_spectral_noise(
    x: np.ndarray,
    mean: float = 0.0,
    noise_std: float = 0.01,
    shift_range: int = 0,
) -> np.ndarray:
    """Add Gaussian noise and optional wavelength shift."""

    x_noisy = x + np.random.normal(mean, noise_std, x.shape)
    if shift_range > 0:
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift > 0:
            x_noisy = np.concatenate([x_noisy[:, shift:], x_noisy[:, -shift:]], axis=1)
        elif shift < 0:
            x_noisy = np.concatenate([x_noisy[:, :shift], x_noisy[:, :-shift]], axis=1)
    return x_noisy


def preprocess_spectral(
    data: np.ndarray,
    crop: Optional[Tuple[int, int]] = (800, 1200),
    derivative: bool = True,
    correction: Optional[str] = "msc",
    smooth_window: int = 50,
    smooth_poly: int = 3,
) -> np.ndarray:
    """Apply the spectral preprocessing path used by the notebook."""

    x = np.asarray(data, dtype=np.float32)
    if derivative:
        x = first_derivative(x)
    if correction == "msc":
        x = msc(x)
    elif correction == "snv":
        x = snv(x)
    elif correction == "baseline":
        x = baseline_correction(x)
    elif correction is not None:
        raise ValueError(f"Unknown correction: {correction}")
    if crop is not None:
        x = x[:, crop[0] : crop[1]]
    return savgol_filter(x, smooth_window, smooth_poly).astype(np.float32)


def load_spectral_csv(
    csv_path: str,
    label_col: str = "label",
    drop_col: Optional[int] = 0,
    test_size: float = 0.2,
    random_state: int = 98,
    batch_size: int = 32,
    train_correction: str = "msc",
    test_correction: str = "snv",
    crop: Optional[Tuple[int, int]] = (800, 1200),
    drop_last: bool = True,
) -> LoaderBundle:
    """Load a spectral CSV and create train/test dataloaders."""

    df = pd.read_csv(csv_path, encoding="utf-8", skiprows=-1)
    labels = df[label_col].to_numpy(dtype=np.int64)
    features_df = df.drop(columns=[label_col])
    if drop_col is not None and df.columns[drop_col] in features_df.columns:
        features_df = features_df.drop(columns=[df.columns[drop_col]])
    data = features_df.to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
        shuffle=True,
    )
    train_data = preprocess_spectral(x_train, crop=crop, correction=train_correction)
    test_data = preprocess_spectral(x_test, crop=crop, correction=test_correction)
    return make_loaders(train_data, test_data, y_train, y_test, batch_size=batch_size, drop_last=drop_last)


def extract_audio_mfcc(file_path: str, n_mfcc: int = 40) -> np.ndarray:
    audio, sample_rate = librosa.load(file_path, sr=None)
    features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(features.T, axis=0).astype(np.float32)


def load_audio_mnist(
    root_folder_path: str,
    test_size: float = 0.25,
    random_state: int = 8,
    batch_size: int = 32,
    n_mfcc: int = 40,
) -> LoaderBundle:
    """Load AudioMNIST folders and return MFCC dataloaders."""

    rows = []
    for speaker in range(1, 61):
        folder = os.path.join(root_folder_path, f"{speaker:02d}")
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            rows.append((extract_audio_mfcc(file_path, n_mfcc=n_mfcc), int(file_name[0])))
    x = np.array([r[0] for r in rows], dtype=np.float32)
    y = np.array([r[1] for r in rows], dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=True, random_state=random_state, stratify=y
    )
    return make_loaders(x_train, x_test, y_train, y_test, batch_size=batch_size)


def load_urbansound8k(
    metadata_csv: str,
    audio_root: str,
    batch_size: int = 32,
    test_size: float = 0.25,
    random_state: int = 8,
    n_mfcc: int = 40,
) -> LoaderBundle:
    """Load UrbanSound8K metadata and create MFCC dataloaders."""

    df = pd.read_csv(metadata_csv)
    features, labels = [], []
    for _, row in df.iterrows():
        file_name = os.path.join(audio_root, f"fold{row['fold']}", row["slice_file_name"])
        features.append(extract_audio_mfcc(file_name, n_mfcc=n_mfcc))
        labels.append(int(row["classID"]))
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return make_loaders(x_train, x_test, y_train, y_test, batch_size=batch_size)


def load_mitbih(
    train_csv: str,
    test_csv: str,
    batch_size: int = 32,
    upsample_classes: Sequence[int] = (1, 2, 3, 4),
    upsample_n: int = 20000,
) -> LoaderBundle:
    """Load MIT-BIH heartbeat CSVs with the notebook's class balancing."""

    train = pd.read_csv(train_csv, header=None)
    test = pd.read_csv(test_csv, header=None)
    extra = []
    for cls in upsample_classes:
        df_cls = train[train[187] == cls]
        extra.append(resample(df_cls, n_samples=upsample_n, replace=True, random_state=123))
    train_df = pd.concat([train[train[187] == 0], *extra])

    x_train = train_df.loc[:, train_df.columns != 187].values.astype(np.float32)
    y_train = train_df.loc[:, train_df.columns == 187].values.squeeze().astype(np.int64)
    x_test = test.loc[:, test.columns != 187].values.astype(np.float32)
    y_test = test.loc[:, test.columns == 187].values.squeeze().astype(np.int64)
    return make_loaders(x_train, x_test, y_train, y_test, batch_size=batch_size)


def load_eeg_csv(
    csv_path: str,
    target_col: str = "target",
    pca_components: int = 256,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
) -> LoaderBundle:
    """Load EEG tabular data, standardize it, and reduce with PCA."""

    df = pd.read_csv(csv_path)
    x = df.drop(columns=[target_col]).values
    y = df[target_col].values.astype(np.int64)
    x = StandardScaler().fit_transform(np.squeeze(x))
    x = PCA(n_components=pca_components).fit_transform(x).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return make_loaders(x_train, x_test, y_train, y_test, batch_size=batch_size)


def encode_labels(labels: Sequence) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder


def make_loaders(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    drop_last: bool = True,
    num_workers: int = 0,
) -> LoaderBundle:
    train_dataset = ArrayDataset(x_train, y_train)
    test_dataset = ArrayDataset(x_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers
    )
    return LoaderBundle(train_loader, test_loader, x_train, x_test, y_train, y_test)


def tensors_to_loaders(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Compatibility helper mirroring the notebook's tensor conversion cells."""

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)
