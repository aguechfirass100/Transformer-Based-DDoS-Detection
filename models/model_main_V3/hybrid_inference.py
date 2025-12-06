import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
import sys
from tqdm import tqdm
from typing import Tuple, List, Dict

try:
    from train import load_config, CONFIG_PATH
    from models.model_main_V3.hybrid_model import HybridTransformerClassifier
    from models.model_main_V3.dataloader import HybridDataModule
    from models.model_basic.model import TransformerClassifier
    from models.model_multi_attention.model import SimplifiedMultiAttentionTransformer
except ImportError:
    print("Error: Could not import project modules.")
    print("Please ensure this script is run from the project's root directory")
    print("or that the project's root directory is in your PYTHONPATH.")
    sys.exit(1)


CLASS_NAMES_MAP = {
    0: 'DrDoS_DNS',
    1: 'DrDoS_LDAP',
    2: 'DrDoS_NetBIOS',
    3: 'DrDoS_NTP',
    4: 'DrDoS_SNMP',
    5: 'TFTP'
}




def create_inference_folder(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_folder = output_dir / f"inference_{timestamp}"
    inference_folder.mkdir(parents=True, exist_ok=True)
    return inference_folder


@torch.no_grad()
def evaluate_model(model: HybridTransformerClassifier,
                   test_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    print(f"Running evaluation on {len(test_loader.dataset)} samples...")
    for batch in tqdm(test_loader, desc="Evaluating Batches"):
        implicit_features, explicit_features_dict, protocol_ids, labels = batch

        implicit_features = implicit_features.to(device)
        protocol_ids = protocol_ids.to(device)
        explicit_features_dict = {k: v.to(device) for k, v in explicit_features_dict.items()}
        labels = labels.to(device)

        start_time = time.perf_counter()

        outputs = model(implicit_features, explicit_features_dict, protocol_ids)

        end_time = time.perf_counter()

        batch_latency_ms = ((end_time - start_time) / len(labels)) * 1000
        latencies.append(batch_latency_ms)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    avg_latency_ms = np.mean(latencies)

    return all_preds, all_labels, all_probs, avg_latency_ms


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          save_path: Path):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion Matrix saved to: {save_path}")
    except Exception as e:
        print(f"\n✗ Error saving confusion matrix: {e}")
    plt.show()


def save_evaluation_report(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: list,
                           avg_latency_ms: float,
                           ckpt_path: str,
                           test_set_size: int,
                           save_path: Path):

    accuracy = (y_pred == y_true).mean()

    str_class_names = [str(cn) for cn in class_names]

    report_str = classification_report(y_true, y_pred, target_names=str_class_names, zero_division=0)

    with open(save_path, 'w') as f:
        f.write(f"Hybrid Model Evaluation Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint: {ckpt_path}\n")
        f.write(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Set Size:     {test_set_size} samples\n")
        f.write(f"Overall Accuracy:  {accuracy:.4f}\n")
        f.write(f"Avg. Latency:      {avg_latency_ms:.4f} ms per sample\n\n")

        f.write(f"CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(report_str)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("CLASS MAPPING\n")
        f.write("=" * 80 + "\n")
        for i, name in enumerate(class_names):
            f.write(f"Class {i}: {name}\n")

    print(f"✓ Detailed report saved to: {save_path}")


def save_predictions_csv(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_probs: np.ndarray,
                         class_names: list,
                         save_path: Path):

    df = pd.DataFrame()
    df['True_Label_ID'] = y_true
    df['Predicted_Label_ID'] = y_pred

    # Map IDs to names
    df['True_Label_Name'] = [class_names[i] for i in y_true]
    df['Predicted_Label_Name'] = [class_names[i] for i in y_pred]

    df['Is_Correct'] = (df['True_Label_ID'] == df['Predicted_Label_ID'])
    df['Confidence'] = [y_probs[i, pred] for i, pred in enumerate(y_pred)]

    for i, class_name in enumerate(class_names):
        df[f'Prob_{class_name}'] = y_probs[:, i]

    try:
        df.to_csv(save_path, index=False)
        print(f"✓ Raw predictions saved to: {save_path}")
    except Exception as e:
        print(f"\n✗ Error saving predictions CSV: {e}")


def find_checkpoint(config_ckpt_dir: str) -> str:

    checkpoint_dir = Path(config_ckpt_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    last_ckpt = checkpoint_dir / "last.ckpt"
    if not last_ckpt.exists():
        # Fallback: find any .ckpt file
        all_ckpts = list(checkpoint_dir.glob("*.ckpt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
        last_ckpt = max(all_ckpts, key=lambda p: p.stat().st_mtime)
        print(f"Warning: 'last.ckpt' not found. Using most recent: {last_ckpt.name}")

    return str(last_ckpt)


def main(args):
    print("\n" + "=" * 80)
    print("HYBRID MODEL INFERENCE SCRIPT")
    print("=" * 80 + "\n")

    print(f"Loading main config from: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    print("Setting up data module...")
    data_cfg = config['data']
    data_module = HybridDataModule(
        data_path=data_cfg['data_path'],
        feature_groups=config['feature_groups'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        test_size=data_cfg['test_size'],
        val_size=data_cfg['val_size'],
        random_state=config['training']['random_seed']
    )
    data_module.setup()

    test_loader = data_module.test_dataloader()
    test_set_size = len(data_module.test_dataset)

    try:
        numeric_classes = sorted(data_module.label_encoder.classes_)
        # Map numeric classes to string names
        class_names = [CLASS_NAMES_MAP[i] for i in numeric_classes]
    except KeyError as e:
        print(f"✗ FATAL ERROR: Class mismatch. LabelEncoder has class {e} which is not in CLASS_NAMES_MAP.")
        print("Please verify the CLASS_NAMES_MAP at the top of this script.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ FATAL ERROR mapping class names: {e}")
        print(f"LabelEncoder classes found: {data_module.label_encoder.classes_}")
        sys.exit(1)

    print(f"✓ Data loaded: {test_set_size} test samples.")
    print(f"✓ Found {len(class_names)} classes. Mapping:")
    for i, name in zip(numeric_classes, class_names):
        print(f"  Class {i}: {name}")

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        print("No checkpoint specified. Finding 'last.ckpt' in config checkpoint directory...")
        ckpt_path = find_checkpoint(config['training']['checkpoint_dir'])

    print(f"Loading model from: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = HybridTransformerClassifier.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            implicit_model_class=TransformerClassifier,
            explicit_model_class=SimplifiedMultiAttentionTransformer,
            strict=False
        )
        print("✓ Hybrid model loaded successfully.")
    except Exception as e:
        print(f"\n✗ FATAL ERROR loading model: {e}")
        print("Please ensure the correct sub-model classes are imported and paths are correct.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    results_folder = create_inference_folder(output_dir)
    print(f"Results will be saved to: {results_folder}\n")

    all_preds, all_labels, all_probs, avg_latency_ms = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")


    report_path = results_folder / "evaluation_report.txt"
    save_evaluation_report(
        all_labels, all_preds, class_names,
        avg_latency_ms, ckpt_path, test_set_size, report_path
    )

    csv_path = results_folder / "predictions.csv"
    save_predictions_csv(
        all_labels, all_preds, all_probs, class_names, csv_path
    )

    cm_path = results_folder / "confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)

    print("\n" + "=" * 80)
    print("Inference run finished successfully.")
    print("=" * 80 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Model Inference Script")

    parser.add_argument(
        '--ckpt',
        type=str,
        default=r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\checkpoints\hybrid\last.ckpt",
        help="Path to the hybrid model checkpoint (.ckpt). If not provided, 'last.ckpt' from config will be used."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\inference_reports",
        help="Base directory to save the timestamped results folder."
    )

    args = parser.parse_args()
    main(args)