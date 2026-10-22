"""Download and validate the credit card fraud dataset.

Source: Machine Learning Group, Université Libre de Bruxelles (ULB) and Worldline.
Transactions made by European cardholders over two days in September 2013.
Features V1-V28 are PCA components (anonymised for confidentiality); Time and Amount
are original. Licence: Open Database License (ODbL) v1.0.
Dal Pozzolo, A., Caelen, O., Johnson, R. A. and Bontempi, G. (2015). Calibrating
probability with undersampling for unbalanced classification. IEEE SSCI.
"""
import urllib.request
from pathlib import Path

import pandas as pd

MIRROR = ("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/"
          "master/creditcard.csv")
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_AMOUNT = 25_162_590.01


def validate(df: pd.DataFrame) -> None:
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert int(df.Class.sum()) == EXPECTED_FRAUD, f"expected {EXPECTED_FRAUD} frauds"
    assert abs(df.Amount.sum() - EXPECTED_AMOUNT) < 1, "total amount does not match the published file"
    assert list(df.columns) == ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR, path)
    df = pd.read_csv(path)
    validate(df)
    return df
