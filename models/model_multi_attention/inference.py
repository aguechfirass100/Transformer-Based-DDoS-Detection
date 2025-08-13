import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
# from model import MultiAttentionTransformer
from model import SimplifiedMultiAttentionTransformer as MultiAttentionTransformer

from flow_dataset import FlowDataModule


def create_inference_folder():
    """
    Create a timestamped folder for saving inference results
    """
    base_path = Path("C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_multi_attention/inference")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_folder = base_path / timestamp

    # Create the folder (including parent directories if they don't exist)
    inference_folder.mkdir(parents=True, exist_ok=True)

    return inference_folder


def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Comprehensive model evaluation with multiple metrics
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Updated to handle the new data structure: features dict, protocol_ids, labels
            if len(batch) == 3:
                features, protocol_ids, labels = batch

                # Move features dict to device
                features_device = {}
                for key, value in features.items():
                    features_device[key] = value.to(device)

                protocol_ids = protocol_ids.to(device)
                labels = labels.to(device)

                # Forward pass with feature groups
                outputs = model(features_device, protocol_ids)
            else:
                raise ValueError(f"Expected 3 elements in batch (features, protocol_ids, labels), got {len(batch)}")

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot and optionally save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_metrics(y_true, y_pred, y_probs, class_names=None):
    """
    Calculate comprehensive evaluation metrics
    """
    # Basic accuracy
    accuracy = (y_pred == y_true).mean()

    # Get unique classes present in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)

    # Adjust class_names if provided but doesn't match
    if class_names is not None and len(class_names) != n_classes:
        print(f"Warning: {len(class_names)} class names provided, but {n_classes} classes found in data")
        class_names = None

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # AUC-ROC for binary classification only
    auc_roc = None
    if n_classes == 2:
        try:
            auc_roc = roc_auc_score(y_true, y_probs[:, 1])
        except:
            print("Warning: Could not calculate AUC-ROC")

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'auc_roc': auc_roc,
        'n_classes': n_classes,
        'unique_classes': unique_classes
    }


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint file in the directory
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    # Sort by modification time and return the latest
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    return str(latest_ckpt)


def main():
    # Load config
    # config_path = Path('config/config.yaml')
    config_path = 'C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_multi_attention/config/config.yaml'

    # if not config_path.exists():
    if not Path(config_path).exists():
        raise FileNotFoundError("config.yaml not found")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Find checkpoint - either use specific path or find latest
    checkpoint_dir = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_multi_attention/lightning_logs/checkpoints"  # Updated path to match your training script
    try:
        # Try to use the specific checkpoint first - update this path as needed
        checkpoint_path = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_multi_attention/lightning_logs/checkpoints/best-checkpoint-10-0.7177.ckpt"
        if not Path(checkpoint_path).exists():
            print(f"Specific checkpoint not found, looking for latest in {checkpoint_dir}")
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    except FileNotFoundError:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Loading model from: {checkpoint_path}")

    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # Updated to use MultiAttentionTransformer
        model = MultiAttentionTransformer.load_from_checkpoint(
            checkpoint_path,
            config=config,
            map_location=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare data with feature groups
    datamodule = FlowDataModule(
        data_file=config['data']['data_file'],
        feature_groups=config['feature_groups'],  # Added feature groups
        label_column=config['data']['label_column'],
        train_frac=config['data'].get('train_fraction', 0.8),
        val_frac=config['data'].get('val_fraction', 0.1),
        test_frac=config['data'].get('test_fraction', 0.1),
        batch_size=config['training']['batch_size']
    )

    try:
        datamodule.setup()
    except Exception as e:
        print(f"Error setting up data module: {e}")
        return

    test_loader = datamodule.test_dataloader()
    print(f"Test set size: {len(datamodule.test_dataset)} samples")

    # Get actual class names from the data module
    actual_class_names = list(datamodule.get_class_names())
    print(f"Actual classes in data: {actual_class_names}")

    # Convert numpy int64 class names to strings for sklearn compatibility
    class_names = [str(name) for name in actual_class_names]
    print(f"Converted class names: {class_names}")

    # Evaluate model
    print("Running evaluation...")
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device)

    # Create timestamped folder for results
    results_folder = create_inference_folder()
    print(f"Results will be saved to: {results_folder}")

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names)

    # Use actual number of classes found
    n_classes = metrics['n_classes']
    unique_classes = metrics['unique_classes']

    # If we have more/fewer classes than expected, adjust class_names
    if len(class_names) != n_classes:
        print(f"Found {n_classes} classes in data, adjusting class names...")
        class_names = [f'Class_{i}' for i in unique_classes]
    else:
        # Ensure class_names are strings and match the number of unique classes
        class_names = [str(name) for name in class_names[:n_classes]]

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    if metrics['auc_roc']:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

    print(f"\nNumber of classes: {n_classes}")
    print("\nDetailed Classification Report:")
    report = metrics['classification_report']

    # Print per-class metrics
    for i, class_name in enumerate(class_names):
        if str(i) in report:  # sklearn uses string keys for class indices
            class_metrics = report[str(i)]
            print(f"{class_name:>15} - Precision: {class_metrics['precision']:.4f}, "
                  f"Recall: {class_metrics['recall']:.4f}, "
                  f"F1: {class_metrics['f1-score']:.4f}, "
                  f"Support: {class_metrics['support']}")

    # Macro and weighted averages
    print(f"\nMacro Avg   - Precision: {report['macro avg']['precision']:.4f}, "
          f"Recall: {report['macro avg']['recall']:.4f}, "
          f"F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg- Precision: {report['weighted avg']['precision']:.4f}, "
          f"Recall: {report['weighted avg']['recall']:.4f}, "
          f"F1: {report['weighted avg']['f1-score']:.4f}")

    # Plot confusion matrix
    try:
        cm_save_path = results_folder / 'confusion_matrix.png'
        plot_confusion_matrix(all_labels, all_preds, class_names,
                              save_path=str(cm_save_path))
        print(f"\nConfusion matrix saved as '{cm_save_path}'")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    # Save detailed results
    results_file = results_folder / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint: {checkpoint_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test Set Size: {len(datamodule.test_dataset)} samples\n")
        f.write(f"Number of Classes: {n_classes}\n\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        if metrics['auc_roc']:
            f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write("=" * 80 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write(f"\n\nClass Names Used:\n")
        for i, name in enumerate(class_names):
            f.write(f"Class {i}: {name}\n")

    print(f"\nDetailed results saved to '{results_file}'")
    print(f"All results saved in folder: {results_folder}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()