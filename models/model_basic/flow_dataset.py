import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class FlowDataset(Dataset):
    def __init__(self, df, numerical_features, categorical_features, label_column):
        self.df = df.reset_index(drop=True)
        self.num_feats = numerical_features
        self.cat_feats = categorical_features
        self.label_col = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract protocol ID
        protocol = int(row["Protocol"])
        protocol = torch.tensor(protocol, dtype=torch.long)

        # Remove "Protocol" from numeric features
        numeric_feats = [f for f in self.num_feats if f != "Protocol"]

        features = []
        if numeric_feats:
            features.append(row[numeric_feats].values.astype(float))
        if self.cat_feats:
            features.append(row[self.cat_feats].values.astype(float))

        # Combine all features
        if features:
            features = torch.tensor(np.concatenate(features), dtype=torch.float32)
        else:
            features = torch.tensor([], dtype=torch.float32)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return features, protocol, label


class FlowDataModule(pl.LightningDataModule):
    def __init__(self, data_file, label_column,
                 train_frac=0.8, val_frac=0.1, test_frac=0.1, batch_size=32):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_file = data_file
        self.label_col = label_column
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.batch_size = batch_size
        self.num_feats = None
        self.cat_feats = []

    def setup(self, stage=None):
        df = pd.read_parquet(self.data_file)

        self.num_feats = [col for col in df.columns if col not in [self.label_col, "Protocol"]]

        self.cat_feats = []

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("Checking for NaNs or infinite values in the training set...")
        # Check for problematic values
        if not np.isfinite(df[self.num_feats].values).all():
            print("⚠️ Warning: Data contains NaNs or infinite values.")
            df[self.num_feats] = df[self.num_feats].replace([np.inf, -np.inf], np.nan)
            df[self.num_feats] = df[self.num_feats].fillna(0)
            print("✅ Replaced inf/-inf with NaN, and filled NaNs with 0.")

        n_total = len(df)
        n_train = max(int(self.train_frac * n_total), 1)
        n_val = max(int(self.val_frac * n_total), 1)

        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

        scaler = StandardScaler()
        train_df[self.num_feats] = scaler.fit_transform(train_df[self.num_feats])
        val_df[self.num_feats] = scaler.transform(val_df[self.num_feats])
        test_df[self.num_feats] = scaler.transform(test_df[self.num_feats])

        self.train_dataset = FlowDataset(train_df, self.num_feats, self.cat_feats, self.label_col)
        self.val_dataset = FlowDataset(val_df, self.num_feats, self.cat_feats, self.label_col)
        self.test_dataset = FlowDataset(test_df, self.num_feats, self.cat_feats, self.label_col)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


