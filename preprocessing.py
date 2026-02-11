"""
Preprocessing module – ABSOLUTE FINAL SAFE VERSION
Works even if old objects are loaded from cache or pickle
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ThyroidDataPreprocessor:
    """
    Bulletproof preprocessor.
    Self-heals missing attributes.
    """

    def __init__(self):
        self._ensure_attributes()

    # --------------------------------------------------

    def _ensure_attributes(self):
        """Ensure ALL required attributes always exist"""
        if not hasattr(self, "num_imputer"):
            self.num_imputer = SimpleImputer(strategy="median")

        if not hasattr(self, "cat_imputer"):
            self.cat_imputer = SimpleImputer(strategy="most_frequent")

        if not hasattr(self, "scaler"):
            self.scaler = StandardScaler()

        if not hasattr(self, "encoder"):
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        if not hasattr(self, "NUMERIC_COLUMNS"):
            self.NUMERIC_COLUMNS = ["age"]

        if not hasattr(self, "CATEGORICAL_COLUMNS"):
            self.CATEGORICAL_COLUMNS = ["sex"]

        if not hasattr(self, "fitted"):
            self.fitted = False

    # --------------------------------------------------

    def preprocess(self, df: pd.DataFrame, is_training: bool = True):
        # 🔥 SELF-HEAL ON EVERY CALL
        self._ensure_attributes()

        df = df.copy()

        # ---------------- CLEAN ----------------
        df.replace(["?", "NA", "N/A", ""], np.nan, inplace=True)

        uncertainty_flags = []

        # ---------------- NUMERIC ----------------
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.NUMERIC_COLUMNS:
            if is_training or not self.fitted:
                df[self.NUMERIC_COLUMNS] = self.num_imputer.fit_transform(
                    df[self.NUMERIC_COLUMNS]
                )
            else:
                df[self.NUMERIC_COLUMNS] = self.num_imputer.transform(
                    df[self.NUMERIC_COLUMNS]
                )

        # ---------------- CATEGORICAL ----------------
        if self.CATEGORICAL_COLUMNS:
            if is_training or not self.fitted:
                df[self.CATEGORICAL_COLUMNS] = self.cat_imputer.fit_transform(
                    df[self.CATEGORICAL_COLUMNS]
                )
            else:
                df[self.CATEGORICAL_COLUMNS] = self.cat_imputer.transform(
                    df[self.CATEGORICAL_COLUMNS]
                )

        # ---------------- BINARY ----------------
        binary_cols = []
        for col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1, True, False, "t", "f"}):
                binary_cols.append(col)
                df[col] = df[col].map(
                    {"t": 1, "f": 0, True: 1, False: 0}
                ).fillna(0)

        # ---------------- ENCODE ----------------
        encoded_df = pd.DataFrame()
        if self.CATEGORICAL_COLUMNS:
            if is_training or not self.fitted:
                encoded = self.encoder.fit_transform(df[self.CATEGORICAL_COLUMNS])
            else:
                encoded = self.encoder.transform(df[self.CATEGORICAL_COLUMNS])

            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.CATEGORICAL_COLUMNS)
            )

        # ---------------- SCALE ----------------
        scaled_df = pd.DataFrame()
        if self.NUMERIC_COLUMNS:
            if is_training or not self.fitted:
                scaled = self.scaler.fit_transform(df[self.NUMERIC_COLUMNS])
            else:
                scaled = self.scaler.transform(df[self.NUMERIC_COLUMNS])

            scaled_df = pd.DataFrame(scaled, columns=self.NUMERIC_COLUMNS)

        # ---------------- COMBINE ----------------
        final_df = pd.concat(
            [
                scaled_df.reset_index(drop=True),
                encoded_df.reset_index(drop=True),
                df[binary_cols].reset_index(drop=True)
            ],
            axis=1
        )

        # ---------------- METADATA ----------------
        missing_ratio = df.isna().mean().mean()
        if missing_ratio > 0:
            uncertainty_flags.append("Missing values were present and imputed")

        metadata = {
            "uncertainty_flags": uncertainty_flags,
            "data_quality_score": max(0.0, 1.0 - missing_ratio)
        }

        self.fitted = True
        return final_df, metadata
