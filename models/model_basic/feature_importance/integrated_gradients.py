import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class FeatureImportanceExtractor:
    def __init__(self, model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.dataloader = dataloader
        self.device = device

    def gradient_based_importance(self, method='integrated_gradients'):
        """Calculate feature importance using gradient-based methods."""
        all_importances = []

        print(f"Calculating {method} feature importance...")

        with torch.enable_grad():
            for batch in tqdm(self.dataloader):
                features, protocol_ids, labels = batch
                features = features.to(self.device).requires_grad_(True)
                protocol_ids = protocol_ids.to(self.device)
                labels = labels.to(self.device)

                if method == 'integrated_gradients':
                    importance = self._integrated_gradients(features, protocol_ids, labels)

                all_importances.append(importance.cpu().detach().numpy())

        # Average across all batches
        return np.mean(np.vstack(all_importances), axis=0)

    def _integrated_gradients(self, features, protocol_ids, labels, steps=50):
        """Integrated gradients method."""
        baseline = torch.zeros_like(features)
        alphas = torch.linspace(0, 1, steps).to(self.device)
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (features - baseline)
            interpolated.requires_grad_(True)
            logits = self.model(interpolated, protocol_ids)
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, interpolated, create_graph=True)[0]
            gradients.append(grad)

        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (features - baseline) * avg_gradients
        importance = torch.abs(integrated_gradients).mean(dim=0)
        return importance


def load_trained_model(checkpoint_path, config):
    """Load the trained model from checkpoint."""
    from models.model_basic.model import TransformerClassifier

    model = TransformerClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def analyze_feature_importance(checkpoint_path, data_module, config):
    """Compute gradient-based feature importance and ranks."""
    # Load model
    print("Loading trained model...")
    model = load_trained_model(checkpoint_path, config)

    # Setup data
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Get feature names
    feature_names = data_module.num_feats

    # Initialize extractor
    extractor = FeatureImportanceExtractor(model, test_loader)

    # Gradient-based importance
    grad_importances = extractor.gradient_based_importance('integrated_gradients')

    # Create DataFrame with ranks
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Integrated_Gradients': grad_importances
    })
    importance_df['Rank'] = importance_df['Integrated_Gradients'].abs().rank(ascending=False)
    importance_df = importance_df.sort_values('Rank')

    # Save results
    importance_df.to_csv('feature_importance_gradient.csv', index=False)
    print("Feature importance analysis complete!")
    return importance_df


# Example usage:
if __name__ == "__main__":
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

    from models.model_basic.flow_dataset import FlowDataModule
    data_module = FlowDataModule(
        data_file='C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/notebooks/IFT_data_ordered.parquet',
        label_column='Label',
        batch_size=32
    )

    checkpoint_path = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_basic/lightning_logs/checkpoints/best-checkpoint-epoch=22-val_loss=0.0032.ckpt"

    importance_df = analyze_feature_importance(checkpoint_path, data_module, config)
    importance_df.to_csv("integrated_gradients.csv", index=False)
    print(importance_df)

