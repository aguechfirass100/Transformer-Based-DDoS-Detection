# Feature Deduplication - Create Reduced Dataset
# This code creates a dataset with correlation-based feature reduction

import pandas as pd
import numpy as np

input_filename = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/data/subset/01-12/flow-level/combined_subset_12k.parquet"  # or .parquet
df = pd.read_parquet(input_filename)

print("=" * 80)
print("FEATURE DEDUPLICATION - CREATING REDUCED DATASET")
print("=" * 80)

# Define features to keep from each correlation group
# Based on the correlation analysis results
features_to_keep_from_groups = {
    # Perfect correlations - keep one from each pair
    'Group_7': 'Total Fwd Packets',  # Drop: Subflow Fwd Packets
    'Group_3': 'Flow IAT Min',  # Drop: Fwd IAT Min
    'Group_28': 'Fwd PSH Flags',  # Drop: RST Flag Count
    'Group_4': 'Fwd Header Length',  # Drop: Fwd Header Length.1

    # Near-perfect correlations - keep most interpretable
    'Group_14': 'Total Length of Fwd Packets',  # Drop: Subflow Fwd Bytes, act_data_pkt_fwd
    'Group_21': 'Packet Length Mean',  # Drop: 6 other packet length features
    'Group_15': 'Flow Packets/s',  # Drop: Fwd Packets/s

    # Strong correlations - keep most representative
    'Group_19': 'Flow Duration',  # Drop: 8 other flow/IAT timing features
    'Group_10': 'Total Backward Packets',  # Drop: 4 other backward packet features
    'Group_11': 'Packet Length Std',  # Drop: 3 other packet length stats
    'Group_16': 'Bwd IAT Mean',  # Drop: Bwd IAT Std, Bwd IAT Max
    'Group_25': 'Protocol',  # Drop: ACK Flag Count
    'Group_6': 'Active Mean',  # Drop: Active Min
    'Group_18': 'Idle Mean',  # Drop: Idle Min
}

# Define all features to drop (from correlation groups)
features_to_drop = [
    # From Group_7 (perfect correlation)
    'Subflow Fwd Packets',

    # From Group_3 (perfect correlation)
    'Fwd IAT Min',

    # From Group_28 (perfect correlation)
    'RST Flag Count',

    # From Group_4 (perfect correlation)
    'Fwd Header Length.1',

    # From Group_14 (near-perfect correlation)
    'Subflow Fwd Bytes',
    'act_data_pkt_fwd',

    # From Group_21 (near-perfect correlation - packet length features)
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Min Packet Length',
    'Average Packet Size',
    'Avg Fwd Segment Size',

    # From Group_15 (near-perfect correlation)
    'Fwd Packets/s',

    # From Group_19 (strong correlation - flow timing features)
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Idle Max',

    # From Group_10 (strong correlation - backward packet features)
    'Total Length of Bwd Packets',
    'Packet Length Variance',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',

    # From Group_11 (strong correlation)
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Avg Bwd Segment Size',

    # From Group_16 (strong correlation)
    'Bwd IAT Std',
    'Bwd IAT Max',

    # From Group_25 (strong correlation)
    'ACK Flag Count',

    # From Group_6 (strong correlation)
    'Active Min',

    # From Group_18 (strong correlation)
    'Idle Min'
]

print(f"  FEATURE REMOVAL SUMMARY:")
print(f"   • Total features to drop: {len(features_to_drop)}")
print(f"   • Features kept from groups: {len(features_to_keep_from_groups)}")

# Display what we're keeping vs dropping by group
print(f"\n  DETAILED REMOVAL PLAN:")
group_details = {
    'Group_7': {'keep': 'Total Fwd Packets', 'drop': ['Subflow Fwd Packets']},
    'Group_3': {'keep': 'Flow IAT Min', 'drop': ['Fwd IAT Min']},
    'Group_28': {'keep': 'Fwd PSH Flags', 'drop': ['RST Flag Count']},
    'Group_4': {'keep': 'Fwd Header Length', 'drop': ['Fwd Header Length.1']},
    'Group_14': {'keep': 'Total Length of Fwd Packets', 'drop': ['Subflow Fwd Bytes', 'act_data_pkt_fwd']},
    'Group_21': {'keep': 'Packet Length Mean',
                 'drop': ['Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                          'Min Packet Length', 'Average Packet Size', 'Avg Fwd Segment Size']},
    'Group_15': {'keep': 'Flow Packets/s', 'drop': ['Fwd Packets/s']},
    'Group_19': {'keep': 'Flow Duration',
                 'drop': ['Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Total', 'Fwd IAT Mean',
                          'Fwd IAT Std', 'Fwd IAT Max', 'Idle Max']},
    'Group_10': {'keep': 'Total Backward Packets',
                 'drop': ['Total Length of Bwd Packets', 'Packet Length Variance', 'Subflow Bwd Packets',
                          'Subflow Bwd Bytes']},
    'Group_11': {'keep': 'Packet Length Std',
                 'drop': ['Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Avg Bwd Segment Size']},
    'Group_16': {'keep': 'Bwd IAT Mean', 'drop': ['Bwd IAT Std', 'Bwd IAT Max']},
    'Group_25': {'keep': 'Protocol', 'drop': ['ACK Flag Count']},
    'Group_6': {'keep': 'Active Mean', 'drop': ['Active Min']},
    'Group_18': {'keep': 'Idle Mean', 'drop': ['Idle Min']}
}

for group, details in group_details.items():
    print(f"   {group}: Keep '{details['keep']}' | Drop {len(details['drop'])} features")

# Create the reduced dataset
print(f"\n  CREATING REDUCED DATASET:")
print(f"   • Original dataset shape: {df.shape}")

# Check which features to drop actually exist in the dataset
existing_features_to_drop = [f for f in features_to_drop if f in df.columns]
missing_features = [f for f in features_to_drop if f not in df.columns]

print(f"   • Features to drop (existing): {len(existing_features_to_drop)}")
if missing_features:
    print(f"   • Features to drop (missing): {len(missing_features)}")
    print(f"     Missing features: {missing_features}")

# Create reduced dataset by dropping correlated features
df_reduced = df.drop(columns=existing_features_to_drop)

print(f"   • Reduced dataset shape: {df_reduced.shape}")
print(f"   • Features removed: {len(existing_features_to_drop)}")
print(f"   • Features remaining: {df_reduced.shape[1] - 1} (+ 1 Label column)")

# Verify the reduction
original_features = len([col for col in df.columns if col != 'Label'])
remaining_features = len([col for col in df_reduced.columns if col != 'Label'])
reduction_percentage = (1 - remaining_features / original_features) * 100

print(f"\n  REDUCTION SUMMARY:")
print(f"   • Original features: {original_features}")
print(f"   • Remaining features: {remaining_features}")
print(f"   • Reduction: {reduction_percentage:.1f}%")

# Save the reduced dataset
output_filename = 'reduced_dataset_correlation_filtered.parquet'
df_reduced.to_parquet(output_filename, index=False)
print(f"\n  REDUCED DATASET SAVED:")
print(f"   • Filename: {output_filename}")
print(f"   • Format: Parquet (efficient storage)")

# Also save as CSV for easy viewing
csv_filename = 'reduced_dataset_correlation_filtered.csv'
df_reduced.to_csv(csv_filename, index=False)
print(f"   • CSV version: {csv_filename}")

# Create a feature mapping file for reference
feature_mapping = []

# Add kept features from groups
for group, kept_feature in features_to_keep_from_groups.items():
    if kept_feature in df_reduced.columns:
        dropped_from_group = [f for f in group_details[group]['drop'] if f in df.columns]
        feature_mapping.append({
            'group': group,
            'kept_feature': kept_feature,
            'dropped_features': ', '.join(dropped_from_group),
            'reason': 'correlation_group_representative'
        })

# Add singleton features
singleton_features = [col for col in df_reduced.columns
                      if col != 'Label' and col not in features_to_keep_from_groups.values()]

for feature in singleton_features:
    feature_mapping.append({
        'group': 'singleton',
        'kept_feature': feature,
        'dropped_features': 'none',
        'reason': 'no_high_correlations'
    })

# Save feature mapping
mapping_df = pd.DataFrame(feature_mapping)
mapping_df.to_csv('feature_reduction_mapping.csv', index=False)

print(f"   • Feature mapping: feature_reduction_mapping.csv")

# Display final feature list
print(f"\n  FINAL FEATURE LIST ({remaining_features} features):")
feature_list = [col for col in df_reduced.columns if col != 'Label']
for i, feature in enumerate(sorted(feature_list), 1):
    print(f"   {i:2d}. {feature}")

# Verify label distribution is preserved
print(f"\n🏷   LABEL DISTRIBUTION CHECK:")
original_labels = df['Label'].value_counts().sort_index()
reduced_labels = df_reduced['Label'].value_counts().sort_index()

print("Original vs Reduced:")
for label in original_labels.index:
    orig_count = original_labels[label]
    red_count = reduced_labels[label]
    print(f"   Label {label}: {orig_count:,} → {red_count:,} {'✓' if orig_count == red_count else '✗'}")

print(f"\n{'=' * 80}")
print("FEATURE DEDUPLICATION COMPLETED SUCCESSFULLY!")
print(f"Your dataset is now ready for feature importance analysis and final selection.")
print(f"{'=' * 80}")