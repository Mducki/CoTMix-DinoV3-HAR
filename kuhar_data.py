import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


CLASS_NAMES = [
    "Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", "Lay",
    "Lay-stand", "Pick", "Jump", "Push-up", "Sit-up", "Walk",
    "Walk-backward", "Walk-circle", "Run", "Stair-up", "Stair-down",
    "Table-tennis",
]
CLASS_TO_ID = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}


def canonical_class(folder_name):
    s = str(folder_name).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "pushup": "push-up", "situp": "sit-up", "talksit": "talk-sit",
        "talkstand": "talk-stand", "standsit": "stand-sit",
        "laystand": "lay-stand", "walkbackward": "walk-backward",
        "walkcircle": "walk-circle", "stairup": "stair-up",
        "stairdown": "stair-down", "tabletennis": "table-tennis",
    }
    s = aliases.get(s.replace("-", ""), aliases.get(s, s))
    for name in CLASS_NAMES:
        if name.lower() == s:
            return name
    return None


def _numeric_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer, header=None, engine="python", on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.to_numpy(dtype=np.float32)


def read_raw_recording(path_or_buffer, sample_rate=100):
    """
    Read one raw KU-HAR CSV recording.
    8-column format: t_acc, ax, ay, az, t_gyro, gx, gy, gz.
    Also supports a six-column sensor-only CSV.
    Returns [6, T] float32 resampled to 100 Hz.
    """
    arr = _numeric_csv(path_or_buffer)
    if arr.ndim != 2 or arr.shape[0] < 20:
        raise ValueError("CSV không có đủ hàng dữ liệu cảm biến.")
    if arr.shape[1] >= 8:
        t_acc = arr[:, 0].copy()
        acc = arr[:, 1:4].copy()
        t_gyro = arr[:, 4].copy()
        gyro = arr[:, 5:8].copy()
        valid_a = np.isfinite(t_acc) & np.isfinite(acc).all(axis=1)
        valid_g = np.isfinite(t_gyro) & np.isfinite(gyro).all(axis=1)
        t_acc, acc = t_acc[valid_a], acc[valid_a]
        t_gyro, gyro = t_gyro[valid_g], gyro[valid_g]
        if len(t_acc) < 2 or len(t_gyro) < 2:
            raise ValueError("Không đọc được timestamp hoặc giá trị cảm biến.")
        # KU-HAR timestamps are described in milliseconds.
        if np.nanmax(t_acc) - np.nanmin(t_acc) > 100:
            t_acc = t_acc / 1000.0
        if np.nanmax(t_gyro) - np.nanmin(t_gyro) > 100:
            t_gyro = t_gyro / 1000.0
        start = max(float(np.min(t_acc)), float(np.min(t_gyro)))
        end = min(float(np.max(t_acc)), float(np.max(t_gyro)))
        if end <= start:
            # Fallback: resample by row index if timestamps are not aligned.
            n = min(len(acc), len(gyro))
            acc, gyro = acc[:n], gyro[:n]
            old = np.linspace(0, 1, n)
            new_n = max(2, int(n))
            new = np.linspace(0, 1, new_n)
            vals = np.stack(
                [np.interp(new, old, acc[:, i]) for i in range(3)]
                + [np.interp(new, old, gyro[:, i]) for i in range(3)]
            )
        else:
            grid = np.arange(start, end, 1.0 / sample_rate, dtype=np.float64)
            if len(grid) < 2:
                raise ValueError("Bản ghi ngắn hơn mức tối thiểu sau nội suy.")
            vals = np.stack(
                [np.interp(grid, t_acc, acc[:, i]) for i in range(3)]
                + [np.interp(grid, t_gyro, gyro[:, i]) for i in range(3)]
            )
    elif arr.shape[1] >= 6:
        vals0 = arr[:, :6]
        vals0 = vals0[np.isfinite(vals0).all(axis=1)]
        if len(vals0) < 20:
            raise ValueError("CSV 6 kênh có quá ít hàng.")
        vals = vals0.T
    else:
        raise ValueError(f"CSV cần ít nhất 6 cột cảm biến; nhận được {arr.shape[1]}.")

    vals = np.asarray(vals, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return vals


def recording_to_windows(path_or_buffer, window=300, stride=300):
    arr = read_raw_recording(path_or_buffer)
    windows = []
    for start in range(0, arr.shape[1] - window + 1, stride):
        windows.append(arr[:, start:start + window])
    if not windows:
        # Allow a short recording for inference by linearly resampling to one window.
        old = np.linspace(0, 1, arr.shape[1])
        new = np.linspace(0, 1, window)
        arr = np.stack([np.interp(new, old, row) for row in arr]).astype(np.float32)
        windows = [arr]
    return np.stack(windows).astype(np.float32)


def load_raw_domain(root, require_all_classes=True):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Domain folder not found: {root}")
    X, y, file_ids = [], [], []
    found_classes = set()
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = canonical_class(class_dir.name)
        if class_name is None:
            continue
        class_id = CLASS_TO_ID[class_name.lower()]
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            continue
        found_classes.add(class_id)
        for csv_path in csv_files:
            try:
                windows = recording_to_windows(csv_path)
            except Exception as exc:
                print(f"[WARN] Skip {csv_path}: {exc}")
                continue
            X.extend(windows)
            y.extend([class_id] * len(windows))
            file_ids.extend([str(csv_path)] * len(windows))
    if not X:
        raise RuntimeError(f"No valid KU-HAR CSV recordings found under {root}")
    if require_all_classes and found_classes != set(range(len(CLASS_NAMES))):
        missing = [CLASS_NAMES[i] for i in sorted(set(range(len(CLASS_NAMES))) - found_classes)]
        raise RuntimeError(
            f"{root} thiếu lớp: {missing}. Mỗi domain nên có đủ 18 lớp để UDA cùng label space."
        )
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64), np.asarray(file_ids)


def split_target_files(target_dir, test_fraction=0.2, seed=42):
    """Split target recordings by file, not by window, to reduce leakage."""
    root = Path(target_dir)
    train_files, test_files = [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if canonical_class(class_dir.name) is None:
            continue
        files = sorted(class_dir.glob("*.csv"))
        if len(files) < 2:
            raise RuntimeError(
                f"Target class '{class_dir.name}' has {len(files)} CSV file(s). "
                "Need >=2 separate recordings per class to hold out target-test by recording."
            )
        tr, te = train_test_split(
            files, test_size=test_fraction, random_state=seed, shuffle=True
        )
        train_files.extend(tr)
        test_files.extend(te)
    return train_files, test_files


def load_selected_files(files):
    X, y, ids = [], [], []
    for path in files:
        class_name = canonical_class(Path(path).parent.name)
        if class_name is None:
            continue
        label = CLASS_TO_ID[class_name.lower()]
        windows = recording_to_windows(path)
        X.extend(windows)
        y.extend([label] * len(windows))
        ids.extend([str(path)] * len(windows))
    if not X:
        raise RuntimeError("No windows loaded from selected target recordings.")
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64), np.asarray(ids)


def load_subsamples_csv(path):
    """
    KU-HAR Time_domain_subsamples:
    columns 0:1800 = six channels x 300 points;
    column 1800 = class id 0..17;
    column 1801 = length; column 1802 = serial number.
    """
    df = pd.read_csv(path, header=None, engine="python", on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[1800])
    if df.shape[1] < 1801:
        raise ValueError(
            f"Expected at least 1801 columns in Time_domain_subsamples, got {df.shape[1]}"
        )
    values = df.iloc[:, :1800].to_numpy(dtype=np.float32)
    labels = df.iloc[:, 1800].to_numpy(dtype=np.int64)
    valid = np.isfinite(values).all(axis=1) & (labels >= 0) & (labels < len(CLASS_NAMES))
    values, labels = values[valid], labels[valid]
    # The CSV stores each sensor channel in its own consecutive 300-column block.
    X = values.reshape(-1, 6, 300).astype(np.float32)
    return X, labels


class SensorDataset(Dataset):
    def __init__(self, X, y=None, mean=None, std=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
        if mean is not None and std is not None:
            m = torch.as_tensor(mean, dtype=torch.float32).view(1, 6, 1)
            s = torch.as_tensor(std, dtype=torch.float32).view(1, 6, 1).clamp_min(1e-6)
            self.X = (self.X - m) / s

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def source_normalization(X):
    mean = X.mean(axis=(0, 2)).astype(np.float32)
    std = X.std(axis=(0, 2)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def make_loader(X, y, mean, std, batch_size, shuffle, drop_last=False):
    ds = SensorDataset(X, y, mean, std)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=0, pin_memory=torch.cuda.is_available()
    )
