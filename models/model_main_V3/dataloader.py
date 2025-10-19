import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any
import pytorch_lightning as pl


class HybridNetworkFlowDataset(Dataset):
    """
    Dataset class that provides data in both formats:
    1. Implicit format: All features as a single tensor
    2. Explicit format: Features grouped by type (temporal, spatial, protocol, statistical)
    """

    def __init__(self,
                 data_path: str,
                 feature_groups: Dict[str, list],
                 split: str = 'train',
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 scaler: StandardScaler = None,
                 label_encoder: LabelEncoder = None):

        self.feature_groups = feature_groups
        self.split = split

        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

        # Prepare data
        self._prepare_data(test_size, val_size, random_state, scaler, label_encoder)

    def _prepare_data(self, test_size, val_size, random_state, scaler, label_encoder):
        """Prepare and split the data"""

        # Separate features and labels
        X = self.df.drop('Label', axis=1)
        y = self.df['Label']

        # Encode labels
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder = label_encoder
            y_encoded = self.label_encoder.transform(y)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )

        # Select split
        if self.split == 'train':
            X_split, y_split = X_train, y_train
        elif self.split == 'val':
            X_split, y_split = X_val, y_val
        else:  # test
            X_split, y_split = X_test, y_test

        # Normalize features
        if scaler is None and self.split == 'train':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_split)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_split) if scaler else X_split.values

        # Store processed data
        self.features_df = pd.DataFrame(X_scaled, columns=X.columns)
        self.labels = torch.tensor(y_split, dtype=torch.long)

        print(f"{self.split.upper()} set: {self.features_df.shape[0]} samples")
        print(f"Number of classes: {len(np.unique(y_split))}")

        # Prepare feature groups for explicit model
        self._prepare_feature_groups()

    def _prepare_feature_groups(self):
        """Prepare feature indices for each group based on column names"""

        all_columns = list(self.features_df.columns)

        # Create mappings from feature names to indices
        self.feature_indices = {}

        for group_name, feature_names in self.feature_groups.items():
            indices = []
            for feature_name in feature_names:
                if feature_name in all_columns:
                    indices.append(all_columns.index(feature_name))
                else:
                    print(f"Warning: Feature '{feature_name}' not found in columns for group '{group_name}'")

            self.feature_indices[group_name] = indices
            print(f"{group_name.upper()} group: {len(indices)} features")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            implicit_features: All features as tensor [num_features]
            explicit_features_dict: Dict with grouped features
            protocol_ids: Protocol ID for implicit model
            label: Class label
        """

        # Get all features for implicit model
        implicit_features = torch.tensor(
            self.features_df.iloc[idx].values,
            dtype=torch.float32
        )

        # Get protocol ID (assuming 'Protocol' column exists)
        if 'Protocol' in self.features_df.columns:
            protocol_id = torch.tensor(
                self.features_df.iloc[idx]['Protocol'],
                dtype=torch.long
            ).clamp(0, 17)  # Ensure within range
        else:
            protocol_id = torch.tensor(0, dtype=torch.long)  # Default protocol

        # Prepare explicit features by groups
        explicit_features_dict = {}
        for group_name, indices in self.feature_indices.items():
            if len(indices) > 0:
                group_features = torch.tensor(
                    self.features_df.iloc[idx].iloc[indices].values,
                    dtype=torch.float32
                )
            else:
                # If no features in group, create dummy feature
                group_features = torch.zeros(1, dtype=torch.float32)

            explicit_features_dict[group_name] = group_features

        label = self.labels[idx]

        return implicit_features, explicit_features_dict, protocol_id, label


class HybridDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for hybrid model training"""

    def __init__(self,
                 data_path: str,
                 feature_groups: Dict[str, list],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):

        super().__init__()
        self.data_path = data_path
        self.feature_groups = feature_groups
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.scaler = None
        self.label_encoder = None

    def setup(self, stage=None):
        """Setup datasets for each stage"""

        # Create train dataset first to fit scaler and label encoder
        self.train_dataset = HybridNetworkFlowDataset(
            data_path=self.data_path,
            feature_groups=self.feature_groups,
            split='train',
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state
        )

        # Get fitted scaler and label encoder
        self.scaler = self.train_dataset.scaler
        self.label_encoder = self.train_dataset.label_encoder

        # Create val and test datasets using fitted transformers
        self.val_dataset = HybridNetworkFlowDataset(
            data_path=self.data_path,
            feature_groups=self.feature_groups,
            split='val',
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
            scaler=self.scaler,
            label_encoder=self.label_encoder
        )

        self.test_dataset = HybridNetworkFlowDataset(
            data_path=self.data_path,
            feature_groups=self.feature_groups,
            split='test',
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
            scaler=self.scaler,
            label_encoder=self.label_encoder
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_num_classes(self):
        """Get number of unique classes"""
        if hasattr(self, 'train_dataset'):
            return len(self.label_encoder.classes_)
        return None

    def get_num_features(self):
        """Get number of features"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.features_df.shape[1]
        return None


# Utility function for testing the dataloader
def test_dataloader(data_path: str, feature_groups: Dict[str, list]):
    """Test function to verify dataloader works correctly"""

    # Create data module
    dm = HybridDataModule(
        data_path=data_path,
        feature_groups=feature_groups,
        batch_size=4,
        num_workers=0  # Use 0 for debugging
    )

    # Setup
    dm.setup()

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    implicit_features, explicit_features_dict, protocol_ids, labels = batch

    print("\n=== DATALOADER TEST ===")
    print(f"Batch size: {implicit_features.shape[0]}")
    print(f"Implicit features shape: {implicit_features.shape}")
    print(f"Protocol IDs shape: {protocol_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    print("\nExplicit features:")
    for group_name, features in explicit_features_dict.items():
        print(f"  {group_name}: {features.shape}")

    print(f"\nNumber of classes: {dm.get_num_classes()}")
    print(f"Number of features: {dm.get_num_features()}")

    return dm


if __name__ == "__main__":
    feature_groups = {
        'temporal': ['Timestamp', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max', 'Flow IAT Max', 'Flow IAT Mean'],
        'spatial': ['Source IP', 'Destination IP', 'Destination Port'],
        'protocol': ['Protocol', 'ACK Flag Count'],
        'statistical': ['Max Packet Length', 'Fwd Packets/s', 'Fwd Packet Length Std', 'Subflow Fwd Bytes',
                        'min_seg_size_forward', 'Packet Length Std', 'Total Length of Fwd Packets',
                        'Fwd Packet Length Max', 'Init_Win_bytes_forward', 'Fwd Header Length',
                        'Fwd Packet Length Min', 'Average Packet Size', 'Min Packet Length']
    }


    data_file = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/IFT_data_ordered.parquet"
    dm = test_dataloader(data_file, feature_groups)