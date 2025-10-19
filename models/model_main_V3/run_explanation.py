import torch
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from train import load_config, CONFIG_PATH
from models.model_main_V3.hybrid_model import HybridTransformerClassifier
from models.model_main_V3.dataloader import HybridDataModule
from models.model_basic.model import TransformerClassifier
from models.model_multi_attention.model import SimplifiedMultiAttentionTransformer
from explainability import HybridModelExplainer


def analyze_single_samples(explainer, test_dataset, output_base_dir: Path,
                           sample_indices: list = None):
    """
    Analyze individual samples in detail.
    """
    print("\n" + "=" * 80)
    print("SINGLE SAMPLE ANALYSIS MODE")
    print("=" * 80)

    if sample_indices is None:
        # Select diverse samples: one from each class if possible
        sample_indices = []
        for class_idx, class_name in enumerate(explainer.class_names):
            # Find first sample of this class
            class_samples = np.where(test_dataset.labels == class_idx)[0]
            if len(class_samples) > 0:
                sample_indices.append(class_samples[0])

        print(f"\nAuto-selected {len(sample_indices)} samples (one per class)")

    print(f"Analyzing samples: {sample_indices}\n")

    for idx in sample_indices:
        if idx >= len(test_dataset):
            print(f"⚠ Skipping sample {idx} (out of range)")
            continue

        sample_row = test_dataset.features_df.iloc[idx]
        true_label = explainer.class_names[test_dataset.labels[idx]]

        sample_output_dir = output_base_dir / "single_samples" / f"sample_{idx:04d}"

        result = explainer.explain_prediction(
            sample_row,
            sample_idx=idx,
            true_label=true_label,
            output_dir=sample_output_dir,
            save_html=True
        )

        print(f"✓ Sample {idx} analysis complete")
        print(f"  True: {true_label} | Predicted: {result['predicted_class_name']}")
        print(f"  Confidence: {result['confidence']:.2%}\n")


def analyze_batch(explainer, test_dataset, output_base_dir: Path,
                  num_samples: int = 50, balanced: bool = True):
    """
    Analyze a batch of samples for aggregate insights.
    """
    print("\n" + "=" * 80)
    print("BATCH ANALYSIS MODE")
    print("=" * 80 + "\n")

    if balanced:
        # Sample equally from each class
        print(f"Sampling {num_samples} examples (balanced across classes)...")
        samples_per_class = num_samples // len(explainer.class_names)

        selected_indices = []
        for class_idx in range(len(explainer.class_names)):
            class_samples = np.where(test_dataset.labels == class_idx)[0]
            if len(class_samples) > 0:
                selected = np.random.choice(
                    class_samples,
                    size=min(samples_per_class, len(class_samples)),
                    replace=False
                )
                selected_indices.extend(selected)

        batch_df = test_dataset.features_df.iloc[selected_indices]
        class_labels = [explainer.class_names[test_dataset.labels[i]]
                        for i in selected_indices]
    else:
        # Random sampling
        print(f"Sampling {num_samples} random examples...")
        batch_df = test_dataset.features_df.sample(n=num_samples, random_state=42)
        class_labels = None

    batch_output_dir = output_base_dir / "batch_analysis"

    result = explainer.explain_batch(
        batch_df,
        output_dir=batch_output_dir,
        class_labels=class_labels
    )

    print(f"✓ Batch analysis complete for {len(batch_df)} samples")


def compare_attack_types(explainer, test_dataset, output_base_dir: Path):
    """
    Compare feature importance across different attack types.
    """
    print("\n" + "=" * 80)
    print("CROSS-ATTACK COMPARISON MODE")
    print("=" * 80 + "\n")

    print("Selecting representative samples from each attack class...")

    # Select one sample from each class
    comparison_indices = []
    for class_idx, class_name in enumerate(explainer.class_names):
        class_samples = np.where(test_dataset.labels == class_idx)[0]
        if len(class_samples) > 0:
            # Select a sample with high confidence (if possible)
            sample_idx = class_samples[np.random.randint(0, len(class_samples))]
            comparison_indices.append(sample_idx)
            print(f"  • {class_name}: Sample {sample_idx}")

    comparison_output_dir = output_base_dir / "attack_comparison"

    explainer.compare_samples(
        comparison_indices,
        test_dataset,
        output_dir=comparison_output_dir
    )

    print(f"✓ Cross-attack comparison complete")


def analyze_misclassifications(explainer, test_dataset, model, device,
                               output_base_dir: Path, num_samples: int = 10):
    """
    Focus analysis on misclassified samples to understand model failures.
    """
    print("\n" + "=" * 80)
    print("MISCLASSIFICATION ANALYSIS MODE")
    print("=" * 80 + "\n")

    print("Finding misclassified samples...")

    # Get predictions for test set (in batches to avoid memory issues)
    batch_size = 256
    all_predictions = []
    all_true_labels = []

    for i in range(0, len(test_dataset), batch_size):
        batch_end = min(i + batch_size, len(test_dataset))
        batch_features = []

        for j in range(i, batch_end):
            sample_row = test_dataset.features_df.iloc[j]
            batch_features.append(sample_row.values)

        batch_np = np.array(batch_features)
        batch_preds = explainer.predict_wrapper(batch_np)
        all_predictions.extend(np.argmax(batch_preds, axis=1))
        all_true_labels.extend(test_dataset.labels[i:batch_end])

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Find misclassifications
    misclassified_indices = np.where(all_predictions != all_true_labels)[0]

    print(f"✓ Found {len(misclassified_indices)} misclassified samples "
          f"({len(misclassified_indices) / len(test_dataset) * 100:.2f}% error rate)")

    if len(misclassified_indices) == 0:
        print("⚠ No misclassifications found!")
        return

    # Sample from misclassifications
    num_to_analyze = min(num_samples, len(misclassified_indices))
    selected_misclass = np.random.choice(misclassified_indices,
                                         size=num_to_analyze,
                                         replace=False)

    print(f"\nAnalyzing {num_to_analyze} misclassified samples in detail...\n")

    misclass_output_dir = output_base_dir / "misclassifications"

    for idx in selected_misclass:
        sample_row = test_dataset.features_df.iloc[idx]
        true_label = explainer.class_names[all_true_labels[idx]]
        pred_label = explainer.class_names[all_predictions[idx]]

        sample_output_dir = misclass_output_dir / f"sample_{idx:04d}_true_{true_label}_pred_{pred_label}"

        result = explainer.explain_prediction(
            sample_row,
            sample_idx=idx,
            true_label=true_label,
            output_dir=sample_output_dir,
            save_html=True
        )

        print(f"✓ Sample {idx}: {true_label} → {pred_label} (Confidence: {result['confidence']:.2%})")


def generate_summary_report(output_base_dir: Path, explainer):
    """
    Generate a comprehensive summary of all analyses.
    """
    summary_path = output_base_dir / "MASTER_SUMMARY.txt"

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MASTER EXPLAINABILITY SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_base_dir}\n\n")

        f.write("-" * 80 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Number of Features:  {len(explainer.feature_names)}\n")
        f.write(f"Number of Classes:   {len(explainer.class_names)}\n")
        f.write(f"Attack Classes:      {', '.join(explainer.class_names)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FEATURE GROUPS\n")
        f.write("-" * 80 + "\n\n")

        for group_name, indices in explainer.feature_indices.items():
            f.write(f"• {group_name.upper()}: {len(indices)} features\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("ANALYSIS MODULES EXECUTED\n")
        f.write("-" * 80 + "\n\n")

        if (output_base_dir / "single_samples").exists():
            num_single = len(list((output_base_dir / "single_samples").iterdir()))
            f.write(f"✓ Single Sample Analysis:     {num_single} samples\n")

        if (output_base_dir / "batch_analysis").exists():
            f.write(f"✓ Batch Analysis:              Completed\n")

        if (output_base_dir / "attack_comparison").exists():
            f.write(f"✓ Cross-Attack Comparison:     Completed\n")

        if (output_base_dir / "misclassifications").exists():
            num_misclass = len(list((output_base_dir / "misclassifications").iterdir()))
            f.write(f"✓ Misclassification Analysis:  {num_misclass} samples\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("HOW TO USE THESE REPORTS\n")
        f.write("-" * 80 + "\n\n")

        f.write("1. SINGLE SAMPLE REPORTS: Review individual predictions in detail\n")
        f.write("   - Check feature_importance.png for top influential features\n")
        f.write("   - Review group_analysis.png for feature group impacts\n")
        f.write("   - Read EXPLANATION_REPORT.txt for textual insights\n\n")

        f.write("2. BATCH ANALYSIS: Understand model behavior across many samples\n")
        f.write("   - batch_summary_bar.png shows overall feature importance\n")
        f.write("   - batch_summary_beeswarm.png shows feature value distributions\n")
        f.write("   - per_class_importance.png compares features across attacks\n\n")

        f.write("3. ATTACK COMPARISON: See how different attacks are distinguished\n")
        f.write("   - sample_comparison.png shows side-by-side feature importance\n\n")

        f.write("4. MISCLASSIFICATION ANALYSIS: Learn where the model struggles\n")
        f.write("   - Focus on these to improve model performance\n")
        f.write("   - Look for patterns in confused attack types\n\n")

    print(f"\n✓ Master summary saved to: {summary_path}")


def main(ckpt_path: str,
         output_dir: str,
         mode: str = "all",
         sample_indices: list = None,
         num_batch: int = 50,
         num_misclass: int = 10):
    """
    Main function to orchestrate explainability analysis.

    Args:
        ckpt_path: Path to trained model checkpoint
        output_dir: Base directory for all outputs
        mode: Analysis mode - 'all', 'single', 'batch', 'compare', or 'misclass'
        sample_indices: Specific samples for single analysis
        num_batch: Number of samples for batch analysis
        num_misclass: Number of misclassifications to analyze
    """

    print("\n" + "=" * 80)
    print("PROFESSIONAL EXPLAINABILITY SUITE")
    print("=" * 80)
    print(f"\nConfiguration: {CONFIG_PATH}")
    print(f"Mode: {mode.upper()}")
    print(f"Output Directory: {output_dir}\n")

    config = load_config(CONFIG_PATH)

    # 1. Setup DataModule
    print("Setting up data module...")
    data_module = HybridDataModule(
        data_path=config['data']['data_path'],
        feature_groups=config['feature_groups'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['training']['random_seed']
    )
    data_module.setup()
    print(f"✓ Test dataset: {len(data_module.test_dataset)} samples\n")

    # 2. Load Trained Model
    if not Path(ckpt_path).exists():
        print(f"✗ Error: Checkpoint not found at {ckpt_path}")
        print("Please train the model first using train.py")
        return

    print(f"Loading model from: {ckpt_path}")
    model = HybridTransformerClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        implicit_model_class=TransformerClassifier,
        explicit_model_class=SimplifiedMultiAttentionTransformer,
        strict=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}\n")

    # 3. Initialize Explainer
    explainer = HybridModelExplainer(model, data_module, background_samples=50)

    output_base_dir = Path(output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 4. Run selected analysis modes
    if mode in ["all", "single"]:
        analyze_single_samples(explainer, data_module.test_dataset,
                               output_base_dir, sample_indices)

    if mode in ["all", "batch"]:
        analyze_batch(explainer, data_module.test_dataset,
                      output_base_dir, num_samples=num_batch, balanced=True)

    if mode in ["all", "compare"]:
        compare_attack_types(explainer, data_module.test_dataset, output_base_dir)

    if mode in ["all", "misclass"]:
        analyze_misclassifications(explainer, data_module.test_dataset,
                                   model, device, output_base_dir,
                                   num_samples=num_misclass)

    # 5. Generate master summary
    generate_summary_report(output_base_dir, explainer)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)
    print(f"\n📁 All reports saved to: {output_base_dir}")
    print(f"📄 Start with MASTER_SUMMARY.txt for navigation guide\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Professional Explainability Suite for Hybrid DDoS Detection Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses
  python run_explanation.py --mode all

  # Analyze specific samples
  python run_explanation.py --mode single --samples 10 25 50 100

  # Batch analysis only
  python run_explanation.py --mode batch --num_batch 100

  # Focus on misclassifications
  python run_explanation.py --mode misclass --num_misclass 20

  # Compare attack types
  python run_explanation.py --mode compare
        """
    )

    DEFAULT_CKPT_PATH = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\checkpoints\hybrid\last.ckpt"

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=DEFAULT_CKPT_PATH,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\shap_reports",
        help='Base directory for all output reports'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'single', 'batch', 'compare', 'misclass'],
        default='all',
        help='Analysis mode to run'
    )
    parser.add_argument(
        '--samples',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample indices for single analysis'
    )
    parser.add_argument(
        '--num_batch',
        type=int,
        default=50,
        help='Number of samples for batch analysis'
    )
    parser.add_argument(
        '--num_misclass',
        type=int,
        default=10,
        help='Number of misclassified samples to analyze'
    )

    args = parser.parse_args()

    main(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        mode=args.mode,
        sample_indices=args.samples,
        num_batch=args.num_batch,
        num_misclass=args.num_misclass
    )