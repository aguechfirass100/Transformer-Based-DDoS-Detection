import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class FeatureImportanceExtractor:
    def __init__(self, model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.dataloader = dataloader
        self.device = device

    def attention_based_importance(self):
        """
        Extract attention weights from transformer layers.
        Note: This requires modifying your model to return attention weights.
        """
        print("⚠️ This method requires model modification to return attention weights")
        print("Consider implementing this in your TransformerClassifier forward method")
        return None

    def gradient_based_importance(self, method='integrated_gradients'):
        """
        Calculate feature importance using gradient-based methods.
        """
        all_importances = []
        all_features = []

        print(f"Calculating {method} feature importance...")

        with torch.enable_grad():
            for batch in tqdm(self.dataloader):
                features, protocol_ids, labels = batch
                features = features.to(self.device).requires_grad_(True)
                protocol_ids = protocol_ids.to(self.device)
                labels = labels.to(self.device)

                if method == 'vanilla_gradients':
                    importance = self._vanilla_gradients(features, protocol_ids, labels)
                elif method == 'integrated_gradients':
                    importance = self._integrated_gradients(features, protocol_ids, labels)
                elif method == 'gradient_x_input':
                    importance = self._gradient_x_input(features, protocol_ids, labels)

                all_importances.append(importance.cpu().detach().numpy())
                all_features.append(features.cpu().detach().numpy())

        return np.vstack(all_importances), np.vstack(all_features)

    def _vanilla_gradients(self, features, protocol_ids, labels):
        """Vanilla gradients method."""
        logits = self.model(features, protocol_ids)
        loss = F.cross_entropy(logits, labels)

        gradients = torch.autograd.grad(loss, features, create_graph=True)[0]
        importance = torch.abs(gradients).mean(dim=0)  # Mean across batch

        return importance

    def _integrated_gradients(self, features, protocol_ids, labels, steps=50):
        """Integrated gradients method."""
        baseline = torch.zeros_like(features)

        # Create path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (features - baseline)
            interpolated.requires_grad_(True)

            logits = self.model(interpolated, protocol_ids)
            loss = F.cross_entropy(logits, labels)

            grad = torch.autograd.grad(loss, interpolated, create_graph=True)[0]
            gradients.append(grad)

        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (features - baseline) * avg_gradients
        importance = torch.abs(integrated_gradients).mean(dim=0)

        return importance

    def _gradient_x_input(self, features, protocol_ids, labels):
        """Gradient × Input method."""
        logits = self.model(features, protocol_ids)
        loss = F.cross_entropy(logits, labels)

        gradients = torch.autograd.grad(loss, features, create_graph=True)[0]
        importance = torch.abs(gradients * features).mean(dim=0)

        return importance

    def permutation_importance(self, n_repeats=10):
        """
        Calculate permutation importance by shuffling each feature.
        """
        print("Calculating permutation importance...")

        # Get baseline accuracy
        baseline_acc = self._evaluate_model()
        print(f"Baseline accuracy: {baseline_acc:.4f}")

        feature_importances = []
        num_features = None

        # Get number of features from first batch
        for batch in self.dataloader:
            features, _, _ = batch
            num_features = features.shape[1]
            break

        for feature_idx in tqdm(range(num_features), desc="Features"):
            importance_scores = []

            for _ in range(n_repeats):
                # Shuffle this feature across all samples
                shuffled_acc = self._evaluate_with_shuffled_feature(feature_idx)
                importance = baseline_acc - shuffled_acc
                importance_scores.append(importance)

            feature_importances.append(np.mean(importance_scores))

        return np.array(feature_importances)

    def _evaluate_model(self):
        """Evaluate model accuracy without modification."""
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dataloader:
                features, protocol_ids, labels = batch
                features = features.to(self.device)
                protocol_ids = protocol_ids.to(self.device)

                logits = self.model(features, protocol_ids)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return accuracy_score(all_labels, all_preds)

    def _evaluate_with_shuffled_feature(self, feature_idx):
        """Evaluate model with one feature shuffled."""
        all_preds = []
        all_labels = []

        # Collect all data first
        all_features = []
        all_protocols = []
        all_true_labels = []

        for batch in self.dataloader:
            features, protocol_ids, labels = batch
            all_features.append(features)
            all_protocols.append(protocol_ids)
            all_true_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_protocols = torch.cat(all_protocols, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)

        # Shuffle the specific feature
        shuffled_features = all_features.clone()
        perm_idx = torch.randperm(len(shuffled_features))
        shuffled_features[:, feature_idx] = shuffled_features[perm_idx, feature_idx]

        # Evaluate with shuffled feature
        with torch.no_grad():
            batch_size = self.dataloader.batch_size
            for i in range(0, len(shuffled_features), batch_size):
                end_idx = min(i + batch_size, len(shuffled_features))

                batch_features = shuffled_features[i:end_idx].to(self.device)
                batch_protocols = all_protocols[i:end_idx].to(self.device)
                batch_labels = all_true_labels[i:end_idx]

                logits = self.model(batch_features, batch_protocols)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        return accuracy_score(all_labels, all_preds)

    def occlusion_importance(self, patch_size=1):
        """
        Calculate importance by occluding (zeroing out) features.
        """
        print("Calculating occlusion-based importance...")

        baseline_acc = self._evaluate_model()
        print(f"Baseline accuracy: {baseline_acc:.4f}")

        # Get number of features
        for batch in self.dataloader:
            features, _, _ = batch
            num_features = features.shape[1]
            break

        feature_importances = []

        for feature_idx in tqdm(range(num_features), desc="Features"):
            occluded_acc = self._evaluate_with_occluded_feature(feature_idx, patch_size)
            importance = baseline_acc - occluded_acc
            feature_importances.append(importance)

        return np.array(feature_importances)

    def _evaluate_with_occluded_feature(self, feature_idx, patch_size=1):
        """Evaluate model with feature(s) occluded (set to 0)."""
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dataloader:
                features, protocol_ids, labels = batch

                # Occlude feature(s)
                occluded_features = features.clone()
                start_idx = max(0, feature_idx - patch_size // 2)
                end_idx = min(features.shape[1], feature_idx + patch_size // 2 + 1)
                occluded_features[:, start_idx:end_idx] = 0

                occluded_features = occluded_features.to(self.device)
                protocol_ids = protocol_ids.to(self.device)

                logits = self.model(occluded_features, protocol_ids)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return accuracy_score(all_labels, all_preds)


def load_trained_model(checkpoint_path, config):
    """Load the trained model from checkpoint."""
    from models.model_basic.model import TransformerClassifier

    model = TransformerClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def plot_feature_importance(importances, feature_names, method_name, top_k=20):
    """Plot feature importance."""
    # Sort features by importance
    sorted_idx = np.argsort(np.abs(importances))[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # Plot top_k features
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(min(top_k, len(sorted_names)))

    plt.barh(y_pos, sorted_importances[:top_k])
    plt.yticks(y_pos, sorted_names[:top_k])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Feature Importance - {method_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return sorted_idx, sorted_importances


def analyze_feature_importance(checkpoint_path, data_module, config):
    """Complete feature importance analysis."""

    # Load model
    print("Loading trained model...")
    model = load_trained_model(checkpoint_path, config)

    # Setup data
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Get feature names (excluding Protocol which is handled separately)
    feature_names = data_module.num_feats

    print(f"Analyzing {len(feature_names)} features...")

    # Initialize extractor
    extractor = FeatureImportanceExtractor(model, test_loader)

    results = {}

    # Method 1: Gradient-based importance
    print("\n" + "=" * 50)
    print("1. INTEGRATED GRADIENTS")
    print("=" * 50)
    grad_importances, _ = extractor.gradient_based_importance('integrated_gradients')
    results['integrated_gradients'] = np.mean(grad_importances, axis=0)
    plot_feature_importance(results['integrated_gradients'], feature_names, 'Integrated Gradients')

    # Method 2: Permutation importance
    print("\n" + "=" * 50)
    print("2. PERMUTATION IMPORTANCE")
    print("=" * 50)
    perm_importance = extractor.permutation_importance(n_repeats=5)
    results['permutation'] = perm_importance
    plot_feature_importance(results['permutation'], feature_names, 'Permutation Importance')

    # Method 3: Occlusion importance
    print("\n" + "=" * 50)
    print("3. OCCLUSION IMPORTANCE")
    print("=" * 50)
    occlusion_importance = extractor.occlusion_importance()
    results['occlusion'] = occlusion_importance
    plot_feature_importance(results['occlusion'], feature_names, 'Occlusion Importance')

    # Create summary DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Integrated_Gradients': results['integrated_gradients'],
        'Permutation': results['permutation'],
        'Occlusion': results['occlusion']
    })

    # Add ranking columns
    for method in ['Integrated_Gradients', 'Permutation', 'Occlusion']:
        importance_df[f'{method}_Rank'] = importance_df[method].abs().rank(ascending=False)

    # Sort by average rank
    importance_df['Avg_Rank'] = importance_df[['Integrated_Gradients_Rank',
                                               'Permutation_Rank',
                                               'Occlusion_Rank']].mean(axis=1)

    importance_df = importance_df.sort_values('Avg_Rank')

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 50)
    print(importance_df.head(20).to_string(index=False))

    return importance_df, results


# Example usage:
if __name__ == "__main__":
    # Your configuration dictionary
    config = {
        'model': {
            'embed_dim': 64,
            'num_heads': 4,
            'num_layers': 3,
            'ffn_dim': 128,
            'dropout': 0.2,
            'num_classes': 6
        },
        'training': {
            'learning_rate': 0.001,
            'epochs': 100
        }
    }

    # Initialize your data module
    from models.model_basic.flow_dataset import FlowDataModule
    data_module = FlowDataModule(
        data_file='C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/IFT_data_ordered.parquet',
        label_column='Label',
        batch_size=32
    )

    # Run analysis
    checkpoint_path = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_basic/lightning_logs/checkpoints/best-checkpoint-epoch=22-val_loss=0.0032.ckpt"

    importance_df, results = analyze_feature_importance(checkpoint_path, data_module, config)

    # Save results
    importance_df.to_csv('feature_importance_analysis.csv', index=False)

    print("Feature importance analysis complete!")