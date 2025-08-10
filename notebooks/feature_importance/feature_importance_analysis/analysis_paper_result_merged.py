import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# File paths
global_feature_file = r"C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/feature_importance/global_feature_importance/xgboost_feature_importance_gain.csv"
per_class_folder = r"C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/feature_importance/per_class_feature_importance/per_class_feature_importance_results"

# Paper's top 5 features per class (from Random Forest)
paper_features = {
    'UDPLag': [
        ('ACK Flag Count', 0.125438),
        ('Init Win bytes forward', 0.002093),
        ('min seg size forward', 0.000795),
        ('Fwd IAT Mean', 0.000612),
        ('Fwd IAT Max', 0.000471)
    ],
    'TFTP': [
        ('Fwd IAT Mean', 0.000207),
        ('min seg size forward', 0.000198),
        ('Fwd IAT Max', 0.000151),
        ('Flow IAT Max', 0.000129),
        ('Flow IAT Mean', 0.000124)
    ],
    'WebDDoS': [
        ('ACK Flag Count', 0.043991),
        ('Init Win bytes forward', 0.009357),
        ('Fwd Packet Length Std', 0.002881),
        ('Packet Length Std', 0.002068),
        ('min seg size forward', 0.000872)
    ],
    'DrDoS_DNS': [
        ('Max Packet Length', 1.139858),
        ('Fwd Packet Length Max', 0.127708),
        ('Fwd Packet Length Min', 0.007794),
        ('Average Packet Size', 0.005849),
        ('Min Packet Length', 0.003487)
    ],
    'benign': [
        ('ACK Flag Count', 0.020021),
        ('Flow IAT Min', 0.016769),
        ('Init Win bytes forward', 0.003182),
        ('Fwd Packet Length Std', 0.001786),
        ('Packet Length Std', 0.001678)
    ],
    'DrDoS_MSSQL': [
        ('Fwd Packets/s', 0.000204),
        ('Protocol', 4.60E-05),
        ('', 0), ('', 0), ('', 0)  # Only 2 features listed
    ],
    'DrDoS_LDAP': [
        ('Max Packet Length', 1.278323),
        ('Fwd Packet Length Max', 0.143219),
        ('Fwd Packet Length Min', 0.008736),
        ('Average Packet Size', 0.006532),
        ('Min Packet Length', 0.003909)
    ],
    'DrDoS_NetBIOS': [
        ('Fwd Packets/s', 0.000172),
        ('min seg size forward', 7.20E-05),
        ('Protocol', 4.60E-05),
        ('Fwd Header Length', 3.50E-05),
        ('Fwd Header Length.1', 3.20E-05)
    ],
    'DrDoS_NTP': [
        ('Subflow Fwd Bytes', 0.106481),
        ('Total Length of Fwd Packets', 0.058022),
        ('Fwd Packet Length Std', 0.001081),
        ('min seg size forward', 0.000707),
        ('Flow IAT Min', 0.000573)
    ],
    'DrDoS_SSDP': [
        ('Destination Port', 0.000671),
        ('Fwd Packet Length Std', 0.000597),
        ('Packet Length Std', 0.000232),
        ('Protocol', 4.60E-05),
        ('min seg size forward', 1.20E-05)
    ],
    'DrDoS_SNMP': [
        ('Max Packet Length', 1.152048),
        ('Fwd Packet Length Max', 0.129074),
        ('Fwd Packet Length Min', 0.007879),
        ('Average Packet Size', 0.005912),
        ('Min Packet Length', 0.003526)
    ],
    'Syn': [
        ('ACK Flag Count', 0.145834),
        ('Init Win bytes forward', 0.002432),
        ('min seg size forward', 0.000872),
        ('Fwd IAT Total', 0.000571),
        ('Flow Duration', 0.000409)
    ],
    'DrDoS_UDP': [
        ('Destination Port', 0.000699),
        ('Fwd Packet Length Std', 0.000615),
        ('Packet Length Std', 0.000239),
        ('min seg size forward', 9.80E-05),
        ('Protocol', 4.50E-05)
    ]
}


# Improved feature name normalization
def normalize_feature_name(name):
    """Comprehensive feature name normalization to handle all variations"""
    if pd.isna(name) or name == '' or str(name).strip() == '':
        return None

    # Convert to string and handle comprehensive cleaning
    name = str(name).strip()

    # Remove any leading/trailing whitespace and special characters
    name = re.sub(r'^\s+|\s+$', '', name)

    # Handle empty strings after cleaning
    if not name:
        return None

    # Specific mappings for known variations
    mappings = {
        'Init Win bytes forward': 'Init_Win_bytes_forward',
        'min seg size forward': 'min_seg_size_forward',
        'Total Length of Fwd Packets': 'Total Length of Fwd Packets',
        'Fwd Header Length.1': 'Fwd Header Length',
        # Add more mappings as needed
    }

    # Apply mappings
    normalized = mappings.get(name, name)

    # Additional cleaning: normalize spaces and special characters
    normalized = re.sub(r'\s+', ' ', normalized)  # Replace multiple spaces with single space
    normalized = normalized.strip()

    return normalized if normalized else None


def clean_dataframe_features(df):
    """Clean feature names in a DataFrame"""
    df_clean = df.copy()

    # Clean feature column if it exists
    if 'feature' in df_clean.columns:
        df_clean['feature'] = df_clean['feature'].apply(normalize_feature_name)
        # Remove rows with None/empty features
        df_clean = df_clean[df_clean['feature'].notna()]
        df_clean = df_clean[df_clean['feature'] != '']
        df_clean = df_clean[df_clean['feature'] != 'Unnamed: 0']

    return df_clean


# Load XGBoost results with proper cleaning
print("Loading XGBoost feature importance files...")
global_importance = pd.read_csv(global_feature_file)
global_importance = clean_dataframe_features(global_importance)
print(f"Global feature importance shape after cleaning: {global_importance.shape}")

# Load all per-class feature importance files with cleaning
xgb_per_class = {}
for file in os.listdir(per_class_folder):
    if file.endswith('.csv'):
        class_name = file.replace('_feature_importance.csv', '')
        file_path = os.path.join(per_class_folder, file)
        df = pd.read_csv(file_path)
        df_clean = clean_dataframe_features(df)
        xgb_per_class[class_name] = df_clean
        print(f"Loaded {class_name}: {df_clean.shape[0]} features")


# Create comprehensive feature importance matrix with better handling
def create_comprehensive_matrix(xgb_global, xgb_per_class, paper_features):
    """Combine XGBoost and paper results into a single matrix with proper deduplication"""

    # Get all unique features with proper normalization
    all_features = set()

    # Add XGBoost global features
    for feature in xgb_global['feature'].tolist():
        normalized = normalize_feature_name(feature)
        if normalized:
            all_features.add(normalized)

    # Add XGBoost per-class features
    for class_df in xgb_per_class.values():
        for feature in class_df['feature'].tolist():
            normalized = normalize_feature_name(feature)
            if normalized:
                all_features.add(normalized)

    # Add paper features (normalized)
    for class_features in paper_features.values():
        for feature, _ in class_features:
            normalized = normalize_feature_name(feature)
            if normalized:
                all_features.add(normalized)

    print(f"\nTotal unique features after normalization: {len(all_features)}")

    # Initialize importance matrix
    feature_list = sorted(list(all_features))
    importance_matrix = pd.DataFrame(index=feature_list)

    # Add XGBoost global importance (normalized)
    xgb_global_clean = xgb_global.copy()
    # Handle potential duplicates by summing importance scores
    xgb_global_agg = xgb_global_clean.groupby('feature')['importance'].sum().reset_index()
    xgb_global_agg['importance'] = xgb_global_agg['importance'] / xgb_global_agg['importance'].sum()

    # Map to matrix
    global_mapping = xgb_global_agg.set_index('feature')['importance']
    importance_matrix['xgb_global'] = importance_matrix.index.map(global_mapping).fillna(0)

    # Add XGBoost per-class importance (normalized and aggregated)
    for class_name, class_df in xgb_per_class.items():
        class_clean = class_df.copy()
        # Aggregate duplicates
        class_agg = class_clean.groupby('feature')['importance'].sum().reset_index()
        class_agg['importance'] = class_agg['importance'] / class_agg['importance'].sum()

        # Map to matrix
        class_mapping = class_agg.set_index('feature')['importance']
        col_name = f'xgb_{class_name}'
        importance_matrix[col_name] = importance_matrix.index.map(class_mapping).fillna(0)

    # Add paper per-class importance (normalized per class)
    for class_name, features in paper_features.items():
        paper_col_name = f'paper_{class_name}'
        importance_matrix[paper_col_name] = 0.0

        # Calculate total importance for normalization
        total_importance = sum(imp for _, imp in features if imp > 0)

        if total_importance > 0:
            for feature, importance in features:
                normalized_feature = normalize_feature_name(feature)
                if normalized_feature and normalized_feature in importance_matrix.index:
                    importance_matrix.loc[normalized_feature, paper_col_name] = importance / total_importance

    return importance_matrix


# Calculate combined scores (same as before but with cleaner data)
def calculate_combined_scores(importance_matrix):
    """Calculate various combined importance scores"""

    # Separate XGBoost and paper columns
    xgb_cols = [col for col in importance_matrix.columns if col.startswith('xgb_')]
    paper_cols = [col for col in importance_matrix.columns if col.startswith('paper_')]

    # Calculate XGBoost average (global weight = 0.4, per-class weight = 0.6)
    per_class_cols = [col for col in xgb_cols if col != 'xgb_global']
    if per_class_cols:
        importance_matrix['xgb_weighted'] = (
                importance_matrix['xgb_global'] * 0.4 +
                importance_matrix[per_class_cols].mean(axis=1) * 0.6
        )
    else:
        importance_matrix['xgb_weighted'] = importance_matrix['xgb_global']

    # Calculate paper average
    if paper_cols:
        importance_matrix['paper_avg'] = importance_matrix[paper_cols].mean(axis=1)
    else:
        importance_matrix['paper_avg'] = 0

    # Combined score (60% XGBoost, 40% paper)
    importance_matrix['combined_score'] = (
            importance_matrix['xgb_weighted'] * 0.6 +
            importance_matrix['paper_avg'] * 0.4
    )

    # Count how many methods/classes consider this feature important
    threshold = 0.01  # Feature is "important" if normalized importance > 1%
    importance_matrix['method_count'] = 0
    importance_matrix['class_count'] = 0

    # Count XGBoost
    importance_matrix['method_count'] += (importance_matrix['xgb_global'] > threshold).astype(int)
    if per_class_cols:
        importance_matrix['class_count'] += (importance_matrix[per_class_cols] > threshold).sum(axis=1)

    # Count paper
    if paper_cols:
        importance_matrix['method_count'] += (importance_matrix[paper_cols] > threshold).any(axis=1).astype(int)
        importance_matrix['class_count'] += (importance_matrix[paper_cols] > threshold).sum(axis=1)

    # Consistency score (features important across multiple classes/methods)
    max_methods = 2  # XGBoost and Paper
    max_classes = len(per_class_cols + paper_cols) if per_class_cols else len(paper_cols)

    if max_classes > 0:
        importance_matrix['consistency_score'] = (
                importance_matrix['method_count'] / max_methods * 0.3 +
                importance_matrix['class_count'] / max_classes * 0.7
        )
    else:
        importance_matrix['consistency_score'] = 0

    # Final score combining importance and consistency
    importance_matrix['final_score'] = (
            importance_matrix['combined_score'] * 0.7 +
            importance_matrix['consistency_score'] * 0.3
    )

    return importance_matrix


# Find optimal number of features using elbow method
def find_elbow_point(importance_matrix, score_column='final_score', max_features=50):
    """Find elbow point in cumulative importance curve"""
    sorted_importance = importance_matrix[score_column].sort_values(ascending=False)
    cumulative_importance = sorted_importance.cumsum() / sorted_importance.sum()

    # Calculate elbow points for different thresholds
    thresholds = [0.80, 0.85, 0.90, 0.95]
    elbow_points = {}

    for threshold in thresholds:
        idx = np.where(cumulative_importance.values >= threshold)[0]
        if len(idx) > 0:
            elbow_points[threshold] = min(idx[0] + 1, len(cumulative_importance))

    return cumulative_importance, elbow_points


# Main analysis
print("\nCreating comprehensive feature importance matrix...")
importance_matrix = create_comprehensive_matrix(global_importance, xgb_per_class, paper_features)
importance_matrix = calculate_combined_scores(importance_matrix)

# Find optimal number of features
cumulative_importance, elbow_points = find_elbow_point(importance_matrix)

print("\n=== Feature Selection Analysis ===")
print(f"Total unique features: {len(importance_matrix)}")
print(f"\nElbow points (cumulative importance):")
for threshold, n_features in elbow_points.items():
    print(f"  {threshold * 100}% importance: {n_features} features")

# Create output directory
output_dir = 'feature_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# Save different feature sets
print(f"\n💾 SAVING FEATURE SETS...")

# 1. All features ranked by final score
all_features_ranked = importance_matrix.sort_values('final_score', ascending=False).copy()
all_features_ranked['rank'] = range(1, len(all_features_ranked) + 1)
all_features_ranked.to_csv(f'{output_dir}/all_features_ranked_by_final_score.csv')

# 2. Recommended features (optimal number)
optimal_count = elbow_points.get(0.9, elbow_points.get(0.85, 30))
recommended_features = all_features_ranked.head(optimal_count)
recommended_features.to_csv(f'{output_dir}/recommended_features_top_{optimal_count}.csv')

# 3. High importance features (top 20)
high_importance = all_features_ranked.head(20)
high_importance.to_csv(f'{output_dir}/high_importance_features_top_20.csv')

# 4. Consistent features (high consistency score)
consistent_features = importance_matrix.sort_values('consistency_score', ascending=False).head(30).copy()
consistent_features['rank'] = range(1, len(consistent_features) + 1)
consistent_features.to_csv(f'{output_dir}/consistent_features_top_30.csv')

# 5. XGBoost specific features
xgb_features = importance_matrix.sort_values('xgb_weighted', ascending=False).head(40).copy()
xgb_features['rank'] = range(1, len(xgb_features) + 1)
xgb_features.to_csv(f'{output_dir}/xgboost_specific_features_top_40.csv')

# 6. Paper specific features
paper_features_df = importance_matrix[importance_matrix['paper_avg'] > 0].sort_values('paper_avg',
                                                                                      ascending=False).copy()
paper_features_df['rank'] = range(1, len(paper_features_df) + 1)
paper_features_df.to_csv(f'{output_dir}/paper_specific_features.csv')

# 7. Feature sets by different thresholds
for threshold, count in elbow_points.items():
    threshold_features = all_features_ranked.head(count)
    threshold_features.to_csv(f'{output_dir}/features_{int(threshold * 100)}percent_threshold_{count}_features.csv')

# 8. Detailed analysis matrix
importance_matrix.to_csv(f'{output_dir}/comprehensive_feature_importance_matrix.csv')

# 9. Feature categories analysis
feature_categories = {
    'Flow_Timing': ['Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min'],
    'Packet_Size': ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
                    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
                    'Packet Length Variance', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'],
    'Flow_Rates': ['Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s'],
    'TCP_Flags': ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                  'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Fwd PSH Flags', 'Bwd PSH Flags',
                  'Fwd URG Flags', 'Bwd URG Flags'],
    'Protocol_Network': ['Source Port', 'Destination Port', 'Protocol', 'Fwd Header Length', 'Bwd Header Length'],
    'Packet_Counts': ['Total Fwd Packets', 'Total Backward Packets', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                      'Subflow Bwd Packets', 'Subflow Bwd Bytes'],
    'Activity_State': ['Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
                       'Idle Min'],
    'TCP_Connection': ['Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward'],
    'Bulk_Transfer': ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                      'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'],
    'Direction_Analysis': ['Down/Up Ratio', 'Inbound'],
    'Special_Features': ['SimillarHTTP']
}

# Save features by category
categories_analysis = []
for category, features in feature_categories.items():
    category_features = [f for f in features if f in importance_matrix.index]
    if category_features:
        category_df = importance_matrix.loc[category_features].copy()
        category_df['category'] = category
        category_df['feature_name'] = category_df.index
        category_df = category_df.sort_values('final_score', ascending=False)
        categories_analysis.append(category_df)

        # Save individual category file
        category_df.to_csv(f'{output_dir}/category_{category.lower()}_features.csv')

if categories_analysis:
    all_categories_df = pd.concat(categories_analysis)
    all_categories_df.to_csv(f'{output_dir}/features_by_categories.csv')

# Create summary report (convert numpy types to native Python types for JSON compatibility)
summary_report = {
    'analysis_summary': {
        'total_features_analyzed': int(len(importance_matrix)),
        'optimal_feature_count': int(optimal_count),
        'elbow_points': {str(k): int(v) for k, v in elbow_points.items()}  # Convert to JSON-serializable types
    },
    'file_descriptions': {
        'all_features_ranked_by_final_score.csv': 'All features ranked by comprehensive final score',
        f'recommended_features_top_{optimal_count}.csv': f'Recommended {optimal_count} features for optimal performance',
        'high_importance_features_top_20.csv': 'Top 20 most important features',
        'consistent_features_top_30.csv': 'Top 30 features with high consistency across classes',
        'xgboost_specific_features_top_40.csv': 'Top 40 features from XGBoost analysis',
        'paper_specific_features.csv': 'Features identified in the research paper',
        'comprehensive_feature_importance_matrix.csv': 'Complete analysis matrix with all scores',
        'features_by_categories.csv': 'Features organized by functional categories'
    }
}

# Save summary
import json

with open(f'{output_dir}/analysis_summary.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

# Create a simple summary CSV
summary_csv_data = []
for i, (feature, row) in enumerate(all_features_ranked.head(50).iterrows(), 1):
    summary_csv_data.append({
        'rank': i,
        'feature': feature,
        'final_score': row['final_score'],
        'xgb_weighted': row['xgb_weighted'],
        'paper_avg': row['paper_avg'],
        'consistency_score': row['consistency_score'],
        'class_count': int(row['class_count']),
        'method_count': int(row['method_count'])
    })

summary_df = pd.DataFrame(summary_csv_data)
summary_df.to_csv(f'{output_dir}/feature_analysis_summary_top_50.csv', index=False)

print(f"   ✅ All results saved to '{output_dir}/' directory:")
print(f"      • {len(os.listdir(output_dir))} files created")
print(f"      • Recommended features: {optimal_count}")
print(f"      • Total features analyzed: {len(importance_matrix)}")

print(f"\n📁 FILES CREATED:")
for file in sorted(os.listdir(output_dir)):
    if file.endswith('.csv'):
        file_path = os.path.join(output_dir, file)
        df_size = len(pd.read_csv(file_path))
        print(f"   • {file:<50} ({df_size} features)")

print(f"\n🎯 TOP 10 RECOMMENDED FEATURES:")
for i, (feature, row) in enumerate(recommended_features.head(10).iterrows(), 1):
    print(f"   {i:2d}. {feature:<40} Score: {row['final_score']:.4f}")

print(f"\n" + "=" * 80)
print("IMPROVED ANALYSIS COMPLETE - Feature name duplicates resolved!")
print("=" * 80)