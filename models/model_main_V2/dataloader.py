import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
import yaml
from typing import Dict, List, Tuple, Any
import random
import warnings
from collections import Counter
import os

warnings.filterwarnings('ignore')


class DDoSFlowDataset(Dataset):
    """Dataset for loading preprocessed DDoS flow data"""

    def __init__(self, flows_data: List[Dict], config: Dict, feature_info: Dict = None,
                 mode: str = 'train', label_encoder=None):
        """
        Args:
            flows_data: List of flow dictionaries
            config: Configuration dictionary
            feature_info: Feature mapping information from preprocessing
            mode: 'train', 'val', or 'test'
            label_encoder: Pre-fitted label encoder (for val/test)
        """
        self.flows_data = flows_data
        self.config = config
        self.feature_info = feature_info
        self.mode = mode
        self.flows_per_batch = config['data']['flows_per_batch']
        self.max_packets = config['data']['max_packets_per_flow']

        # Get packet processing config
        self.packet_config = config.get('packet_processing', {})
        self.padding_strategy = self.packet_config.get('padding_strategy', 'zero')
        self.min_packets = self.packet_config.get('min_packets_per_flow', 1)
        self.remove_all_zero_packets = self.packet_config.get('remove_all_zero_packets', True)

        # Process labels
        labels = [flow['label'] for flow in flows_data]

        # Filter to only keep the 3 DDoS types from config
        valid_classes = config['class_names']
        filtered_flows = []
        filtered_labels = []

        for flow, label in zip(flows_data, labels):
            if label in valid_classes:
                filtered_flows.append(flow)
                filtered_labels.append(label)

        self.flows_data = filtered_flows
        labels = filtered_labels

        if len(self.flows_data) == 0:
            raise ValueError("No valid flows found after filtering! Check your class names.")

        print(f"Filtered to {len(self.flows_data)} flows with valid classes: {valid_classes}")

        # Handle label encoding
        if mode == 'train' or label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)

            # Print class distribution
            unique_labels, counts = np.unique(self.encoded_labels, return_counts=True)
            print(f"\nClass distribution in {mode} set:")
            for label_idx, count in zip(unique_labels, counts):
                label_name = self.label_encoder.inverse_transform([label_idx])[0]
                print(f"  {label_name}: {count} ({count / len(labels) * 100:.1f}%)")
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(labels)

        # Setup feature scaling for flow features
        self.setup_feature_scaling()

        # Group flows by class for balanced sampling
        self.flows_by_class = {}
        for idx, label in enumerate(self.encoded_labels):
            if label not in self.flows_by_class:
                self.flows_by_class[label] = []
            self.flows_by_class[label].append(idx)

        # Print final dimensions
        sample_flow = self.flows_data[0]
        print(f"\nFinal data dimensions:")
        print(f"  Flow features: {len(sample_flow['flow_features'])}")
        print(f"  Packet features: {len(sample_flow['packets'][0]) if sample_flow['packets'] else 0}")
        print(f"  Max packets per flow: {self.max_packets}")

    def setup_feature_scaling(self):
        """Setup feature scaling for flow features"""
        if self.mode == 'train':
            # Extract all flow features for fitting scaler
            all_flow_features = [flow['flow_features'] for flow in self.flows_data]
            all_flow_features = np.array(all_flow_features, dtype=np.float32)

            # Use RobustScaler for better handling of outliers
            self.flow_scaler = RobustScaler()
            self.flow_scaler.fit(all_flow_features)

            # Scale the features
            self.scaled_flow_features = self.flow_scaler.transform(all_flow_features)

            # Also setup packet scaling
            all_packets = []
            for flow in self.flows_data:
                packets = flow['packets']
                if packets:
                    packets_np = np.array(packets, dtype=np.float32)
                    non_zero_mask = np.any(packets_np != 0, axis=1)
                    all_packets.extend(packets_np[non_zero_mask])

            if all_packets:
                all_packets = np.array(all_packets, dtype=np.float32)
                self.packet_scaler = RobustScaler()
                self.packet_scaler.fit(all_packets)
            else:
                self.packet_scaler = None

        else:
            # For validation/test, scalers will be passed from train dataset
            self.flow_scaler = None
            self.packet_scaler = None
            self.scaled_flow_features = None

    def set_scalers(self, flow_scaler, packet_scaler):
        """Set scalers from training dataset"""
        self.flow_scaler = flow_scaler
        self.packet_scaler = packet_scaler

        # Scale flow features
        all_flow_features = [flow['flow_features'] for flow in self.flows_data]
        all_flow_features = np.array(all_flow_features, dtype=np.float32)
        self.scaled_flow_features = self.flow_scaler.transform(all_flow_features)

    def process_packets(self, packets: List[List]) -> np.ndarray:
        """
        Process packets to handle padding and scaling

        Args:
            packets: List of packet feature lists

        Returns:
            Processed packet array
        """
        if not packets:
            # Create dummy packet if no packets
            packets = [[0.0] * 25]  # Assume 25 features per packet, adjust if needed

        # Remove all-zero packets if configured
        if self.remove_all_zero_packets:
            non_zero_packets = [p for p in packets if any(x != 0 for x in p)]
            if non_zero_packets:
                packets = non_zero_packets
            else:
                # Keep at least one packet (can be zero)
                packets = [packets[0]] if packets else [[0.0] * 25]

        # Ensure minimum packets per flow
        while len(packets) < self.min_packets:
            if packets:
                if self.padding_strategy == 'repeat_last':
                    packets.append(packets[-1].copy())
                elif self.padding_strategy == 'mean':
                    mean_packet = np.mean(packets, axis=0).tolist()
                    packets.append(mean_packet)
                else:  # zero padding
                    packets.append([0.0] * len(packets[0]))
            else:
                packets.append([0.0] * 25)

        # Convert to numpy array
        packets = np.array(packets, dtype=np.float32)

        # Scale packets if scaler is available
        if self.packet_scaler is not None and len(packets) > 0:
            # Only scale non-zero packets
            non_zero_mask = np.any(packets != 0, axis=1)
            if np.any(non_zero_mask):
                scaled_non_zero = self.packet_scaler.transform(packets[non_zero_mask])
                packets[non_zero_mask] = scaled_non_zero

        # Handle NaN and inf values
        packets = np.nan_to_num(packets, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad or truncate to max_packets
        current_len = len(packets)
        if current_len > self.max_packets:
            packets = packets[:self.max_packets]
        elif current_len < self.max_packets:
            padding_needed = self.max_packets - current_len
            packet_dim = packets.shape[1]
            padding = np.zeros((padding_needed, packet_dim), dtype=np.float32)
            packets = np.vstack([packets, padding])

        return packets

    def __len__(self):
        return len(self.flows_data) // self.flows_per_batch + (1 if len(self.flows_data) % self.flows_per_batch > 0 else 0)

    def __getitem__(self, idx):
        """Returns a batch of flows"""
        # Sample flows for this batch
        if self.mode == 'train':
            flow_indices = self._sample_balanced_flows()
        else:
            start_idx = idx * self.flows_per_batch
            end_idx = min(start_idx + self.flows_per_batch, len(self.flows_data))
            flow_indices = list(range(start_idx, end_idx))
            # Pad if necessary for val
            while len(flow_indices) < self.flows_per_batch:
                flow_indices.append(flow_indices[0] if flow_indices else 0)

        # Prepare batch data
        batch_packets = []
        batch_flow_features = []
        batch_labels = []
        batch_mask = []
        batch_packet_masks = []  # Track valid packets

        for flow_idx in flow_indices:
            flow = self.flows_data[flow_idx]

            # Get scaled flow features
            scaled_features = self.scaled_flow_features[flow_idx]
            batch_flow_features.append(scaled_features)

            # Process packets
            packets_np = self.process_packets(flow['packets'])

            # Create packet mask: 1 for valid (non-padding), 0 for padding
            num_valid_packets = min(len(flow['packets']), self.max_packets)
            packet_mask = np.ones(self.max_packets, dtype=np.float32)
            packet_mask[num_valid_packets:] = 0.0
            batch_packet_masks.append(packet_mask)

            batch_packets.append(packets_np)

            batch_labels.append(self.encoded_labels[flow_idx])

            # Flow mask: 1 if valid flow, but since we sample valid ones, always 1
            batch_mask.append(1.0)

        # Convert to tensors
        batch_packets = np.array(batch_packets, dtype=np.float32)
        batch_flow_features = np.array(batch_flow_features, dtype=np.float32)
        batch_labels = np.array(batch_labels, dtype=np.int64)
        batch_mask = np.array(batch_mask, dtype=np.float32)
        batch_packet_masks = np.array(batch_packet_masks, dtype=np.float32)

        return {
            'packets': torch.from_numpy(batch_packets),
            'flow_features': torch.from_numpy(batch_flow_features),
            'labels': torch.from_numpy(batch_labels),
            'mask': torch.from_numpy(batch_mask),
            'packet_masks': torch.from_numpy(batch_packet_masks)
        }

    def _sample_balanced_flows(self):
        """Enhanced balanced sampling with better class distribution"""
        num_classes = len(self.flows_by_class)
        if num_classes == 0:
            raise ValueError("No classes found!")

        flows_per_class = self.flows_per_batch // num_classes
        remaining = self.flows_per_batch % num_classes

        sampled_indices = []
        for class_label, indices in self.flows_by_class.items():
            n_samples = flows_per_class + (1 if class_label < remaining else 0)
            if len(indices) == 0:
                continue
            sampled = random.choices(indices, k=n_samples) if n_samples > len(indices) else random.sample(indices, n_samples)
            sampled_indices.extend(sampled)

        # If still short, sample randomly
        while len(sampled_indices) < self.flows_per_batch:
            random_class = random.choice(list(self.flows_by_class.keys()))
            if self.flows_by_class[random_class]:
                sampled_indices.append(random.choice(self.flows_by_class[random_class]))

        random.shuffle(sampled_indices)
        return sampled_indices[:self.flows_per_batch]


def load_flows_data(pkl_path: str) -> Tuple[List[Dict], Dict]:
    """Load flows data and feature info from pickle files"""
    print(f"Loading flows data from {pkl_path}")

    # Load main flows data
    with open(pkl_path, 'rb') as f:
        flows_dict = pickle.load(f)

    flows_data = list(flows_dict.values())  # Directly use values as list of dicts

    print(f"Loaded {len(flows_data)} flows")

    # Load feature info from the same directory
    pkl_dir = os.path.dirname(pkl_path)
    # feature_info_path = os.path.join(pkl_dir, 'pruned/pruned_feature_info.pkl')
    feature_info_path = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main\data\original_data\finale_12_classes\pruned\pruned_feature_info.pkl"
    feature_info = {}

    try:
        with open(feature_info_path, 'rb') as f:
            feature_info = pickle.load(f)
        print(f"✓ Loaded feature info from {feature_info_path}")
        print(f"  Feature columns: {len(feature_info.get('feature_columns', []))}")
    except FileNotFoundError:
        print(f"Warning: Feature info not found at {feature_info_path}")

    return flows_data, feature_info


def create_feature_indices_from_config(config: Dict, feature_info: Dict) -> Dict:
    """
    Convert feature names from config to indices using feature_info

    Returns:
        feature_indices: {head_name: [start_idx, end_idx]} ready for model
    """
    if not feature_info or 'feature_columns' not in feature_info:
        print("Warning: No feature_info available, using fallback indices")
        return None

    feature_columns = feature_info['feature_columns']
    config_features = config['preprocessing']['attention_head_features']

    # Create name to index mapping
    name_to_idx = {name: idx for idx, name in enumerate(feature_columns)}

    feature_indices = {}
    used_indices = set()

    print("Mapping feature names to indices:")

    for head_name, feature_names in config_features.items():
        if head_name == 'feature':
            continue  # Skip 'feature' - we'll handle it last

        indices = []
        found_features = []

        for feature_name in feature_names:
            if feature_name in name_to_idx:
                idx = name_to_idx[feature_name]
                indices.append(idx)
                used_indices.add(idx)
                found_features.append(feature_name)
            else:
                print(f"  Warning: Feature '{feature_name}' not found in data")

        if indices:
            # Sort indices to get contiguous ranges if possible
            indices.sort()
            feature_indices[head_name] = [min(indices), max(indices) + 1]
            print(f"  {head_name}: {len(found_features)} features -> indices {feature_indices[head_name]}")
        else:
            print(f"  {head_name}: No valid features found")

    # Add remaining features to 'feature' head
    remaining_indices = [i for i in range(len(feature_columns)) if i not in used_indices]
    if remaining_indices:
        remaining_indices.sort()
        feature_indices['feature'] = [min(remaining_indices), max(remaining_indices) + 1]
        print(f"  feature: {len(remaining_indices)} remaining features -> indices {feature_indices['feature']}")

    return feature_indices


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, LabelEncoder, Dict]:
    """Create data loaders for preprocessed data"""

    # Load data
    flows_data, feature_info = load_flows_data(config['data']['pkl_file_path'])

    # Create feature indices mapping from config + feature_info
    feature_indices = create_feature_indices_from_config(config, feature_info)

    # Add the feature indices to feature_info for the model
    if feature_indices:
        feature_info['feature_indices'] = feature_indices

    # Split data
    labels = [flow['label'] for flow in flows_data]
    label_counts = Counter(labels)
    min_count = min(label_counts.values()) if label_counts else 0

    if min_count < 2 or not config['validation']['stratify']:
        print("Warning: Some classes have very few samples or stratify disabled. Using simple train_test_split.")
        train_flows, val_flows = train_test_split(
            flows_data,
            test_size=config['validation']['val_split'],
            random_state=config['validation']['random_state'],
            shuffle=True
        )
    else:
        # Use stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config['validation']['val_split'],
            random_state=config['validation']['random_state']
        )

        train_idx, val_idx = next(sss.split(flows_data, labels))
        train_flows = [flows_data[i] for i in train_idx]
        val_flows = [flows_data[i] for i in val_idx]

    print(f"Train flows: {len(train_flows)}, Validation flows: {len(val_flows)}")

    # Create datasets
    train_dataset = DDoSFlowDataset(train_flows, config, feature_info, mode='train')

    val_dataset = DDoSFlowDataset(
        val_flows, config, feature_info, mode='val',
        label_encoder=train_dataset.label_encoder
    )

    # Pass scalers from train to validation dataset
    val_dataset.set_scalers(train_dataset.flow_scaler, train_dataset.packet_scaler)

    # Update config with dims from data
    config['data']['flow_features_dim'] = len(train_flows[0]['flow_features'])
    config['data']['packet_features_dim'] = len(train_flows[0]['packets'][0]) if train_flows[0]['packets'] else 25

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory'],
        drop_last=False
    )

    return train_loader, val_loader, train_dataset.label_encoder, feature_info


def get_class_weights(dataset: DDoSFlowDataset) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    class_counts = np.bincount(dataset.encoded_labels)
    total_samples = len(dataset.encoded_labels)

    # Use inverse frequency weighting with smoothing
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)

    # Cap maximum weight to prevent extreme imbalance
    max_weight = 10.0
    class_weights = np.minimum(class_weights, max_weight)

    return torch.FloatTensor(class_weights)


if __name__ == "__main__":
    # Test the dataloader
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    try:
        train_loader, val_loader, label_encoder, feature_info = create_data_loaders(config)

        print(f"\nDataloader created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Classes: {list(label_encoder.classes_)}")

        # Test one batch
        for batch in train_loader:
            print(f"\nSample batch:")
            print(f"  Packets shape: {batch['packets'].shape}")
            print(f"  Flow features shape: {batch['flow_features'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Mask shape: {batch['mask'].shape}")
            print(f"  Packet masks shape: {batch['packet_masks'].shape}")

            # Check for problematic values
            print(f"\nData quality check:")
            print(f"  Packets - NaN: {torch.isnan(batch['packets']).sum().item()}, Inf: {torch.isinf(batch['packets']).sum().item()}")
            print(f"  Flow features - NaN: {torch.isnan(batch['flow_features']).sum().item()}, Inf: {torch.isinf(batch['flow_features']).sum().item()}")

            # Print packet statistics
            non_zero_packets = (batch['packets'].sum(dim=-1) != 0).sum(dim=-1)
            print(f"  Non-zero packets per flow: {non_zero_packets}")

            break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()