import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class FlowDataset(Dataset):
    def __init__(self, df, feature_groups, label_column):
        """
        Args:
            df: Input DataFrame
            feature_groups: Dict of feature lists for each attention type:
                {
                    'temporal': [...],
                    'spatial': [...],
                    'protocol': [...],
                    'statistical': [...]
                }
            label_column: Name of target column
        """
        self.df = df.reset_index(drop=True)
        self.feature_groups = feature_groups
        self.label_col = label_column

        # Validate all features exist in DataFrame
        all_features = sum(feature_groups.values(), [])
        missing = set(all_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features in DataFrame: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract grouped features
        features = {
            'temporal': torch.tensor(row[self.feature_groups['temporal']].values, dtype=torch.float32),
            'spatial': torch.tensor(row[self.feature_groups['spatial']].values, dtype=torch.float32),
            'protocol': torch.tensor(row[self.feature_groups['protocol']].values, dtype=torch.float32),
            'statistical': torch.tensor(row[self.feature_groups['statistical']].values, dtype=torch.float32)
        }

        # Protocol ID (for embedding)
        protocol_id = torch.tensor(int(row["Protocol"]), dtype=torch.long)

        # Label (already encoded by DataModule)
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return features, protocol_id, label


class FlowDataModule(pl.LightningDataModule):
    def __init__(self, data_file, feature_groups, label_column,
                 train_frac=0.8, val_frac=0.1, test_frac=0.1, batch_size=128):
        super().__init__()
        self.data_file = data_file
        self.feature_groups = feature_groups
        self.label_col = label_column
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.batch_size = batch_size

        # Will be set during setup()
        self.scalers = {}  # One scaler per feature group
        self.label_encoder = LabelEncoder()  # Add label encoder
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        df = pd.read_parquet(self.data_file)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Debug: Check original labels
        print(f"Original labels: {sorted(df[self.label_col].unique())}")
        print(f"Label counts: {df[self.label_col].value_counts().to_dict()}")

        # Encode labels to ensure they're in range [0, num_classes-1]
        df[self.label_col] = self.label_encoder.fit_transform(df[self.label_col])

        # Debug: Check encoded labels
        print(f"Encoded labels: {sorted(df[self.label_col].unique())}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Clean data
        for group, features in self.feature_groups.items():
            if not np.isfinite(df[features].values).all():
                print(f"  Cleaning NaNs/inf in {group} features")
                df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Train/val/test split
        n_total = len(df)
        n_train = max(int(self.train_frac * n_total), 1)
        n_val = max(int(self.val_frac * n_total), 1)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        # Fit scalers on training data only
        self.scalers = {
            group: StandardScaler().fit(train_df[features])
            for group, features in self.feature_groups.items()
        }

        # Scale features
        def scale_features(df):
            scaled_df = df.copy()
            for group, scaler in self.scalers.items():
                scaled_df[self.feature_groups[group]] = scaler.transform(
                    scaled_df[self.feature_groups[group]]
                )
            return scaled_df

        self.train_dataset = FlowDataset(scale_features(train_df), self.feature_groups, self.label_col)
        self.val_dataset = FlowDataset(scale_features(val_df), self.feature_groups, self.label_col)
        self.test_dataset = FlowDataset(scale_features(test_df), self.feature_groups, self.label_col)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def get_num_classes(self):
        """Return the actual number of classes in the dataset"""
        return len(self.label_encoder.classes_)

    def get_class_names(self):
        """Return the original class names"""
        return self.label_encoder.classes_