import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
global_feature_file = r"C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/feature_importance/global_feature_importance/xgboost_feature_importance_gain.csv"
per_class_folder = r"C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/feature_importance/per_class_feature_importance/per_class_feature_importance_results"

# Load global feature importance
global_importance = pd.read_csv(global_feature_file)
print(f"Global feature importance shape: {global_importance.shape}")

# Load all per-class feature importance files
per_class_importance = {}
for file in os.listdir(per_class_folder):
    if file.endswith('.csv'):
        class_name = file.replace('.csv', '')
        file_path = os.path.join(per_class_folder, file)
        per_class_importance[class_name] = pd.read_csv(file_path)
        print(f"Loaded {class_name}: {per_class_importance[class_name].shape}")


# Feature selection strategies

def get_top_k_features(df, k=30, importance_col='importance'):
    """Get top k features based on importance"""
    return df.nlargest(k, importance_col)['feature'].tolist()


def get_features_above_threshold(df, threshold=0.01, importance_col='importance'):
    """Get features above a certain importance threshold"""
    # Normalize importance scores
    df['normalized_importance'] = df[importance_col] / df[importance_col].sum()
    return df[df['normalized_importance'] > threshold]['feature'].tolist()


def get_cumulative_importance_features(df, cumulative_threshold=0.95, importance_col='importance'):
    """Get features that contribute to cumulative_threshold of total importance"""
    df_sorted = df.sort_values(importance_col, ascending=False).copy()
    df_sorted['cumulative_importance'] = df_sorted[importance_col].cumsum() / df_sorted[importance_col].sum()
    return df_sorted[df_sorted['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()


# Strategy 1: Union of top features from each class
def strategy_union_top_k(global_df, per_class_dict, k=10):
    """Take union of top k features from global and each class"""
    all_features = set(get_top_k_features(global_df, k))

    for class_name, class_df in per_class_dict.items():
        class_features = get_top_k_features(class_df, k)
        all_features.update(class_features)

    return list(all_features)


# Strategy 2: Weighted importance across all classes
def strategy_weighted_importance(global_df, per_class_dict, global_weight=0.5):
    """Combine global and per-class importance with weights"""
    # Create a comprehensive feature importance matrix
    all_features = global_df['feature'].tolist()

    # Initialize importance matrix
    importance_matrix = pd.DataFrame(index=all_features)

    # Add global importance (normalized)
    global_norm = global_df.copy()
    global_norm['importance'] = global_norm['importance'] / global_norm['importance'].sum()
    importance_matrix['global'] = global_norm.set_index('feature')['importance']

    # Add per-class importance (normalized)
    for class_name, class_df in per_class_dict.items():
        class_norm = class_df.copy()
        class_norm['importance'] = class_norm['importance'] / class_norm['importance'].sum()
        importance_matrix[class_name] = class_norm.set_index('feature')['importance']

    # Fill NaN values with 0
    importance_matrix = importance_matrix.fillna(0)

    # Calculate weighted average
    per_class_weight = (1 - global_weight) / len(per_class_dict)
    importance_matrix['weighted_avg'] = (
            importance_matrix['global'] * global_weight +
            importance_matrix[list(per_class_dict.keys())].sum(axis=1) * per_class_weight
    )

    return importance_matrix


# Strategy 3: Features that are important for multiple classes
def strategy_multi_class_important(global_df, per_class_dict, min_classes=3, top_k=20):
    """Select features that appear in top k for at least min_classes"""
    feature_counts = {}

    # Count appearances in top k for each class
    for class_name, class_df in per_class_dict.items():
        top_features = get_top_k_features(class_df, top_k)
        for feature in top_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Also consider global top features
    global_top = get_top_k_features(global_df, top_k)
    for feature in global_top:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Select features that appear in at least min_classes
    selected_features = [f for f, count in feature_counts.items() if count >= min_classes]

    return selected_features, feature_counts


# Apply different strategies
print("\n=== Feature Selection Results ===")

# Strategy 1: Union of top k
top_k = 15
union_features = strategy_union_top_k(global_importance, per_class_importance, k=top_k)
print(f"\nStrategy 1 - Union of top {top_k} features: {len(union_features)} features selected")

# Strategy 2: Weighted importance
importance_matrix = strategy_weighted_importance(global_importance, per_class_importance, global_weight=0.4)
weighted_top_features = importance_matrix.nlargest(30, 'weighted_avg').index.tolist()
print(f"\nStrategy 2 - Weighted importance top 30: {len(weighted_top_features)} features selected")

# Strategy 3: Multi-class important features
multi_class_features, feature_counts = strategy_multi_class_important(
    global_importance, per_class_importance, min_classes=4, top_k=20
)
print(f"\nStrategy 3 - Features important for 4+ classes: {len(multi_class_features)} features selected")

# Strategy 4: Cumulative importance
cumulative_features = get_cumulative_importance_features(global_importance, cumulative_threshold=0.9)
print(f"\nStrategy 4 - Cumulative 90% importance: {len(cumulative_features)} features selected")

# Find consensus features (appear in multiple strategies)
all_strategies = [
    set(union_features),
    set(weighted_top_features),
    set(multi_class_features),
    set(cumulative_features)
]

# Features that appear in at least 2 strategies
consensus_features = []
for feature in set().union(*all_strategies):
    count = sum(1 for s in all_strategies if feature in s)
    if count >= 2:
        consensus_features.append(feature)

print(f"\n=== Consensus Features (appear in 2+ strategies): {len(consensus_features)} features ===")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Global feature importance
ax1 = axes[0, 0]
top_global = global_importance.nlargest(20, 'importance')
ax1.barh(top_global['feature'], top_global['importance'])
ax1.set_xlabel('Importance')
ax1.set_title('Top 20 Global Features')
ax1.invert_yaxis()

# Plot 2: Feature appearance across classes
ax2 = axes[0, 1]
feature_appearance = pd.Series(feature_counts).sort_values(ascending=False).head(20)
ax2.bar(range(len(feature_appearance)), feature_appearance.values)
ax2.set_xticks(range(len(feature_appearance)))
ax2.set_xticklabels(feature_appearance.index, rotation=45, ha='right')
ax2.set_ylabel('Number of Classes')
ax2.set_title('Feature Appearance Across Classes')

# Plot 3: Weighted importance distribution
ax3 = axes[1, 0]
importance_matrix['weighted_avg'].sort_values(ascending=False).head(20).plot(kind='bar', ax=ax3)
ax3.set_xlabel('Feature')
ax3.set_ylabel('Weighted Importance')
ax3.set_title('Top 20 Features by Weighted Importance')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Heatmap of top features across classes
ax4 = axes[1, 1]
top_features_for_heatmap = importance_matrix.nlargest(15, 'weighted_avg').index
heatmap_data = importance_matrix.loc[top_features_for_heatmap, list(per_class_importance.keys())]
sns.heatmap(heatmap_data.T, cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Normalized Importance'})
ax4.set_xlabel('Features')
ax4.set_ylabel('Attack Classes')
ax4.set_title('Feature Importance Across Attack Classes')

plt.tight_layout()
plt.show()

# Final recommendation
print("\n=== RECOMMENDATION ===")
print(f"Based on the analysis, I recommend using the consensus features ({len(consensus_features)} features)")
print("These features are consistently important across multiple selection strategies.")
print("\nIf you need a specific number of features, you can:")
print(f"- Use top 20 features: {len(weighted_top_features[:20])} features")
print(f"- Use top 30 features: {len(weighted_top_features[:30])} features")
print(f"- Use top 40 features: {len(weighted_top_features[:40])} features")

# Save selected features
output_dir = "selected_features"
os.makedirs(output_dir, exist_ok=True)

# Save consensus features
pd.DataFrame({'feature': consensus_features}).to_csv(
    os.path.join(output_dir, 'consensus_features.csv'), index=False
)

# Save weighted top features
pd.DataFrame({'feature': weighted_top_features}).to_csv(
    os.path.join(output_dir, 'weighted_top_30_features.csv'), index=False
)

# Save feature importance matrix
importance_matrix.to_csv(os.path.join(output_dir, 'feature_importance_matrix.csv'))

print(f"\nFeature lists saved to '{output_dir}' directory")


# Function to get final feature list
def get_final_features(n_features=30):
    """Get the final recommended feature list"""
    return weighted_top_features[:n_features]


# Example usage
final_features = get_final_features(25)
print(f"\nFinal selected {len(final_features)} features:")
for i, feature in enumerate(final_features, 1):
    print(f"{i}. {feature}")