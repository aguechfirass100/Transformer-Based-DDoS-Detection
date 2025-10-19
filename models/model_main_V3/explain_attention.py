import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os
import warnings
from typing import Dict, List, Tuple

# --- Import from your project files ---
from train import load_config, CONFIG_PATH
from models.model_main_V3.hybrid_model import HybridTransformerClassifier
from models.model_main_V3.dataloader import HybridDataModule
from models.model_basic.model import TransformerClassifier
from models.model_multi_attention.model import SimplifiedMultiAttentionTransformer

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings("ignore", module="matplotlib")

# Enhanced color schemes for different attention types
ATTENTION_COLORMAPS = {
    'implicit': 'YlOrRd',  # Yellow-Orange-Red for implicit
    'explicit': 'YlGnBu',  # Yellow-Green-Blue for explicit
    'fusion': 'RdPu',  # Red-Purple for fusion
    'spatial': 'BuPu',  # Blue-Purple for spatial
    'temporal': 'Oranges',  # Orange for temporal
}


def plot_attention_heatmap(weights: torch.Tensor,
                           title: str,
                           xticklabels: list,
                           yticklabels: list,
                           save_path: Path,
                           attention_type: str = 'default',
                           show_colorbar: bool = True,
                           annotate: bool = True):
    """
    Visualizes attention weights as an enhanced heatmap with better readability.

    Args:
        weights (torch.Tensor): Attention weights, shape [batch_size, num_heads, query_len, key_len]
        title (str): Title for the plot.
        xticklabels (list): Labels for the X-axis (Keys).
        yticklabels (list): Labels for the Y-axis (Queries).
        save_path (Path): Path to save the PNG file.
        attention_type (str): Type of attention for color scheme selection.
        show_colorbar (bool): Whether to show the colorbar.
        annotate (bool): Whether to annotate cells with values.
    """

    # 1. Detach, move to CPU, and remove batch dim
    weights = weights.squeeze(0).detach().cpu()

    # 2. Average over all attention heads -> [query_len, key_len]
    if weights.dim() == 3:
        weights_avg = weights.mean(dim=0)
    else:
        weights_avg = weights

    # 3. Convert to numpy for plotting
    weights_np = weights_avg.numpy()

    # 4. Determine figure size dynamically
    num_cols = len(xticklabels)
    num_rows = len(yticklabels)

    # Scale based on matrix size
    cell_width = 0.8 if num_cols > 10 else 1.2
    cell_height = 0.6 if num_rows > 10 else 1.0

    width = max(8, num_cols * cell_width)
    height = max(6, num_rows * cell_height)

    # 5. Choose colormap based on attention type
    cmap = ATTENTION_COLORMAPS.get(attention_type, 'viridis')

    # 6. Create plot with enhanced styling
    fig, ax = plt.subplots(figsize=(width, height))

    # Format annotation based on matrix size
    fmt = '.2f' if num_cols * num_rows > 100 else '.3f'
    annot = annotate and (num_cols * num_rows <= 200)  # Don't annotate huge matrices

    sns.heatmap(
        weights_np,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar=show_colorbar,
        square=True,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Attention Weight', 'shrink': 0.8}
    )

    # Enhanced title with better formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Keys (Attending To)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Queries (Attending From)", fontsize=11, fontweight='bold')

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # 7. Add statistics text box
    stats_text = f"Max: {weights_np.max():.3f}\nMin: {weights_np.min():.3f}\nMean: {weights_np.mean():.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # 8. Save figure with high DPI
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {save_path.name}")


def create_summary_report(sample_idx: int,
                          label_name: str,
                          pred_name: str,
                          pred_prob: float,
                          attn_summary: Dict,
                          output_dir: Path):
    """
    Creates a text summary report for the sample.
    """
    report_path = output_dir / f"sample_{sample_idx}_SUMMARY.txt"

    with open(report_path, 'w', encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ATTENTION ANALYSIS REPORT - Sample {sample_idx}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"TRUE LABEL:       {label_name}\n")
        f.write(f"PREDICTED LABEL:  {pred_name}\n")
        f.write(f"CONFIDENCE:       {pred_prob:.2%}\n")
        f.write(f"CORRECT:          {' YES' if label_name == pred_name else ' NO'}\n\n")

        f.write("-" * 80 + "\n")
        f.write("ATTENTION MECHANISMS CAPTURED\n")
        f.write("-" * 80 + "\n\n")

        for mechanism, details in attn_summary.items():
            f.write(f"• {mechanism}:\n")
            for key, value in details.items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")

    print(f" Summary report saved to: {report_path.name}")


def explain_sample_attention(model, datamodule, sample_idx: int, output_dir: Path, device: torch.device):
    """
    Runs a single sample through the model and visualizes all its attention weights
    with enhanced visualizations and summary report.
    """

    # 1. Get the raw sample from the dataset
    test_dataset = datamodule.test_dataset
    implicit_features, explicit_features_dict, protocol_id, label = test_dataset[sample_idx]

    # 2. Get metadata for labels
    label_name = datamodule.label_encoder.classes_[label]

    # --- Get all feature names for labeling ---
    all_feature_names = list(test_dataset.features_df.columns)

    # Get feature names for each explicit group
    group_feature_names = {}
    for group_name, feature_list in datamodule.feature_groups.items():
        indices = test_dataset.feature_indices.get(group_name, [])
        if indices:
            group_feature_names[group_name] = [all_feature_names[i] for i in indices]
        else:
            group_feature_names[group_name] = [group_name]

    # Labels for high-level explicit groups
    explicit_group_labels = ['Temporal', 'Spatial', 'Protocol', 'Statistical']
    fusion_labels = ['Implicit Branch', 'Explicit Branch']

    # 3. Add batch dimension and move to device
    implicit_features = implicit_features.unsqueeze(0).to(device)
    protocol_id = protocol_id.unsqueeze(0).to(device)
    explicit_features_dict = {
        k: v.unsqueeze(0).to(device) for k, v in explicit_features_dict.items()
    }

    # 4. Run forward pass requesting attention weights
    print(f"\n{'=' * 80}")
    print(f"ANALYZING SAMPLE {sample_idx}")
    print(f"{'=' * 80}")
    print(f"True Label: {label_name}")

    with torch.no_grad():
        logits, attn_weights_dict = model(
            implicit_features,
            explicit_features_dict,
            protocol_id,
            return_attn_weights=True
        )

    # 5. Get prediction
    pred_prob = torch.softmax(logits, dim=1).max().item()
    pred_idx = torch.argmax(logits, dim=1).item()
    pred_name = datamodule.label_encoder.classes_[pred_idx]

    print(f"Prediction: {pred_name} (Confidence: {pred_prob:.2%})")
    print(f"Result: {' CORRECT' if label_name == pred_name else ' INCORRECT'}")
    print(f"\nGenerating attention visualizations...\n")

    # Dictionary to store attention summary
    attn_summary = {}

    # --- 6. Plot all available attention weights with enhanced styling ---

    # Plot 1: Implicit Model
    if attn_weights_dict.get('implicit_model_attns'):
        implicit_attns = attn_weights_dict['implicit_model_attns']['transformer_encoder_layers']
        last_layer_attn = implicit_attns[-1]

        # Truncate feature names if too long
        truncated_features = [name[:20] + '...' if len(name) > 20 else name
                              for name in all_feature_names]

        plot_attention_heatmap(
            weights=last_layer_attn,
            title=f"Implicit Model: Feature Self-Attention (Final Layer {len(implicit_attns)})",
            xticklabels=truncated_features,
            yticklabels=truncated_features,
            save_path=output_dir / f"1_implicit_self_attention.png",
            attention_type='implicit'
        )

        attn_summary['Implicit Model'] = {
            'Layers': len(implicit_attns),
            'Features': len(all_feature_names),
            'Attention Type': 'Self-Attention'
        }

    # Plot 2: Explicit Model Sub-groups
    if attn_weights_dict.get('explicit_model_attns'):
        explicit_attns = attn_weights_dict['explicit_model_attns']
        attn_summary['Explicit Model'] = {}

        # Temporal
        if explicit_attns.get('temporal_self_attn') is not None:
            temporal_labels = [f'T{i + 1}' for i in range(8)]
            plot_attention_heatmap(
                weights=explicit_attns['temporal_self_attn'],
                title="Explicit Model: Temporal Features Self-Attention",
                xticklabels=temporal_labels,
                yticklabels=temporal_labels,
                save_path=output_dir / f"2a_explicit_temporal.png",
                attention_type='temporal'
            )
            attn_summary['Explicit Model']['Temporal'] = '8 time-based features'

        # Statistical
        if explicit_attns.get('statistical_self_attn') is not None:
            plot_attention_heatmap(
                weights=explicit_attns['statistical_self_attn'],
                title="Explicit Model: Statistical Features Self-Attention",
                xticklabels=['Statistical'],
                yticklabels=['Statistical'],
                save_path=output_dir / f"2b_explicit_statistical.png",
                attention_type='explicit',
                annotate=False
            )
            attn_summary['Explicit Model']['Statistical'] = '1 aggregated feature'

        # Protocol
        if explicit_attns.get('protocol_self_attn') is not None:
            plot_attention_heatmap(
                weights=explicit_attns['protocol_self_attn'],
                title="Explicit Model: Protocol Features Self-Attention",
                xticklabels=['Protocol'],
                yticklabels=['Protocol'],
                save_path=output_dir / f"2c_explicit_protocol.png",
                attention_type='explicit',
                annotate=False
            )
            attn_summary['Explicit Model']['Protocol'] = '1 protocol identifier'

        # Spatial (bidirectional cross-attention)
        if explicit_attns.get('spatial_cross_attn') is not None:
            src_attn, dst_attn = explicit_attns['spatial_cross_attn']
            plot_attention_heatmap(
                weights=src_attn,
                title="Explicit Model: Spatial Cross-Attention (Source → Destination)",
                xticklabels=['Destination'],
                yticklabels=['Source'],
                save_path=output_dir / f"2d_explicit_spatial_src_to_dst.png",
                attention_type='spatial'
            )
            plot_attention_heatmap(
                weights=dst_attn,
                title="Explicit Model: Spatial Cross-Attention (Destination → Source)",
                xticklabels=['Source'],
                yticklabels=['Destination'],
                save_path=output_dir / f"2e_explicit_spatial_dst_to_src.png",
                attention_type='spatial'
            )
            attn_summary['Explicit Model']['Spatial'] = 'Bidirectional (Source ↔ Dest)'

        # Feature group cross-attention
        if explicit_attns.get('feature_group_cross_attn') is not None:
            plot_attention_heatmap(
                weights=explicit_attns['feature_group_cross_attn'],
                title="Explicit Model: Inter-Group Cross-Attention",
                xticklabels=explicit_group_labels,
                yticklabels=explicit_group_labels,
                save_path=output_dir / f"3_explicit_group_cross_attention.png",
                attention_type='explicit'
            )
            attn_summary['Explicit Model']['Group Interaction'] = '4x4 cross-attention'

        # Transformer layers
        if explicit_attns.get('transformer_encoder_layers') is not None:
            last_layer_attn = explicit_attns['transformer_encoder_layers'][-1]
            plot_attention_heatmap(
                weights=last_layer_attn,
                title=f"Explicit Model: Final Transformer Layer Self-Attention (Layer {len(explicit_attns['transformer_encoder_layers'])})",
                xticklabels=explicit_group_labels,
                yticklabels=explicit_group_labels,
                save_path=output_dir / f"4_explicit_transformer_final.png",
                attention_type='explicit'
            )
            attn_summary['Explicit Model']['Transformer Layers'] = len(explicit_attns['transformer_encoder_layers'])

    # Plot 3: Hybrid Fusion Attention
    if attn_weights_dict.get('fusion_attention') is not None:
        plot_attention_heatmap(
            weights=attn_weights_dict['fusion_attention'],
            title=f"Hybrid Fusion: Branch Integration Attention",
            xticklabels=fusion_labels,
            yticklabels=fusion_labels,
            save_path=output_dir / f"5_fusion_attention.png",
            attention_type='fusion'
        )
        attn_summary['Fusion Layer'] = {
            'Type': 'Cross-Branch Attention',
            'Branches': 'Implicit + Explicit'
        }

    # Create summary report
    create_summary_report(sample_idx, label_name, pred_name, pred_prob, attn_summary, output_dir)

    if not attn_weights_dict:
        print(" No attention weights were captured.")


def main(ckpt_path: str, output_dir: str, sample_indices: List[int] = None):
    """
    Main function to load model, setup dataloader, and run explanations.
    """

    print(f"\n{'=' * 80}")
    print("ATTENTION VISUALIZATION TOOL")
    print(f"{'=' * 80}\n")
    print(f"Configuration: {CONFIG_PATH}")

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
    print(f" Test dataset size: {len(data_module.test_dataset)} samples")

    # 2. Load Trained Model
    if not Path(ckpt_path).exists():
        print(f" Error: Checkpoint file not found at {ckpt_path}")
        return

    print(f"\nLoading model from: {ckpt_path}")
    model = HybridTransformerClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        implicit_model_class=TransformerClassifier,
        explicit_model_class=SimplifiedMultiAttentionTransformer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")

    # 3. Run explanations for samples
    output_path = Path(output_dir)

    if sample_indices is None:
        sample_indices = [10, 50, 100, 200, 500]  # Default samples

    print(f"\nAnalyzing {len(sample_indices)} samples: {sample_indices}")

    for idx in sample_indices:
        if idx >= len(data_module.test_dataset):
            print(f"⚠ Skipping sample {idx} (out of range)")
            continue

        sample_output_dir = output_path / f"sample_{idx:04d}"
        explain_sample_attention(model, data_module, idx, sample_output_dir, device)
        print(f"✓ Complete report saved to: {sample_output_dir}\n")

    print(f"\n{'=' * 80}")
    print(f"ALL REPORTS GENERATED")
    print(f"Output directory: {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Enhanced Attention Visualizations for Hybrid Model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    DEFAULT_CKPT_PATH = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\checkpoints\hybrid\last.ckpt"

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=DEFAULT_CKPT_PATH,
        help='Path to the trained model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\attention_reports",
        help='Directory to save attention heatmaps and reports'
    )
    parser.add_argument(
        '--samples',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample indices to analyze (e.g., --samples 10 50 100)'
    )

    args = parser.parse_args()
    main(args.ckpt_path, args.output_dir, args.samples)