import os
from typing import Dict, Any, Optional, Union, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

from .registry import DATASET_REGISTRY


# PAD-UFES-20 diagnostics (after clustering BOD into SCC): ACK, BCC, MEL, NEV, SCC, SEK
CANCER_SET = {"BCC", "SCC", "MEL", "BOD"}  # include BOD defensively
NON_CANCER_SET = {"ACK", "NEV", "SEK"}


def _normalize_diag(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.upper()
    # common variants
    if s in {"BOWEN", "BOWENS", "BOD"}:
        return "BOD"
    return s


class PADUFES20Dataset(Dataset):
    """PAD-UFES-20 clinical (smartphone) skin lesion dataset.

    Expected metadata columns (per paper):
      - patient_id, lesion_id, img_id, biopsy, diagnostic, ...
    Images are typically PNG named: <img_id>.png
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        img_dir: str = "",
        img_id_col: str = "img_id",
        label_col: str = "diagnostic",
        label_mode: str = "binary_cancer",  # or "multiclass_6"
        transform=None,
        img_ext: str = "png",
        is_train: bool = True,
        class_map: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        if df is None:
            if csv_path is None:
                raise ValueError("Provide either df or csv_path.")
            df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_id_col = img_id_col
        self.label_col = label_col
        self.label_mode = label_mode
        self.transform = transform
        self.img_ext = img_ext
        self.is_train = is_train

        if self.img_id_col not in self.df.columns:
            raise KeyError(f"img_id_col='{self.img_id_col}' not found. Available columns: {list(self.df.columns)}")
        if self.label_col not in self.df.columns:
            raise KeyError(f"label_col='{self.label_col}' not found. Available columns: {list(self.df.columns)}")

        # default 6-class mapping (stable order)
        self.class_map = class_map or {
            "ACK": 0,
            "BCC": 1,
            "MEL": 2,
            "NEV": 3,
            "SCC": 4,
            "SEK": 5,
        }

    def __len__(self) -> int:
        return len(self.df)

    def _img_path(self, img_id: str) -> str:
        # img_id may already include extension
        candidate = os.path.join(self.img_dir, img_id)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(self.img_dir, f"{img_id}.{self.img_ext}")
        if os.path.exists(candidate):
            return candidate
        # some downloads put images in a nested folder
        raise FileNotFoundError(f"Cannot find image for img_id={img_id} under {self.img_dir}")

    def _label(self, diag_raw: Any) -> int:
        diag = _normalize_diag(diag_raw)

        if self.label_mode == "binary_cancer":
            # 1 = malignant (cancer), 0 = benign / non-cancer
            return int(diag in CANCER_SET)
        elif self.label_mode == "multiclass_6":
            # cluster BOD into SCC
            if diag == "BOD":
                diag = "SCC"
            if diag not in self.class_map:
                raise KeyError(f"Unknown diagnostic '{diag}'. Known: {list(self.class_map.keys())}")
            return int(self.class_map[diag])
        else:
            raise ValueError("label_mode must be 'binary_cancer' or 'multiclass_6'")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_id = str(row[self.img_id_col])
        img_path = self._img_path(img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image, self.is_train)

        label = self._label(row[self.label_col])
        return {"pixel_values": image, "label": torch.tensor(label, dtype=torch.long)}


@DATASET_REGISTRY.register()
def load_pad_ufes_20(*args, **kwargs):
    """Build PAD-UFES-20 train/test splits.

    Required kwargs:
      - csv_path: path to metadata.csv
      - img_dir: directory containing images (.png)
      - preprocessor: handled by build_dataset()

    Optional kwargs:
      - img_id_col (default 'img_id')
      - label_col (default 'diagnostic')
      - label_mode: 'binary_cancer' or 'multiclass_6' (default 'binary_cancer')
      - split_by: column used for grouping to avoid leakage (default 'patient_id')
      - test_size: fraction for test split (default 0.2)
      - seed: random seed (default 42)

    Notes:
      - We split by patient_id by default to prevent the same patient appearing in both splits.
    """
    csv_path = kwargs["csv_path"]
    img_dir = kwargs["img_dir"]

    img_id_col = kwargs.get("img_id_col", "img_id")
    label_col = kwargs.get("label_col", "diagnostic")
    label_mode = kwargs.get("label_mode", "binary_cancer")
    img_ext = kwargs.get("img_ext", "png")

    split_by = kwargs.get("split_by", "patient_id")
    test_size = float(kwargs.get("test_size", 0.2))
    seed = int(kwargs.get("seed", 42))

    df = pd.read_csv(csv_path)

    if split_by not in df.columns:
        raise KeyError(f"split_by='{split_by}' not found. Available columns: {list(df.columns)}")

    # Group split (patient-wise) to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    y = df[label_col] if label_col in df.columns else None
    groups = df[split_by]
    train_idx, test_idx = next(gss.split(df, y=y, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    common = dict(
        img_dir=img_dir,
        img_id_col=img_id_col,
        label_col=label_col,
        label_mode=label_mode,
        img_ext=img_ext,
        transform=kwargs.get("transform", None),
        class_map=kwargs.get("class_map", None),
    )

    dataset = {
        "train": PADUFES20Dataset(df=train_df, is_train=True, **common),
        "test": PADUFES20Dataset(df=test_df, is_train=False, **common),
    }
    return dataset
