import torch

import shap

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

import warnings

from datetime import datetime

from typing import Dict, List, Tuple, Optional

import json

warnings.filterwarnings("ignore", module="matplotlib")

# Professional color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#C73E1D',
    'neutral': '#6C757D'
}


class HybridModelExplainer:
    """
    Professional explainability suite for Hybrid Transformer models.
    Provides SHAP-based interpretations with comprehensive visualizations.
    """

    def __init__(self, model, datamodule, background_samples: int = 100):
        print("\\n" + "=" * 80)
        print("INITIALIZING PROFESSIONAL EXPLAINABILITY SUITE")
        print("=" * 80 + "\\n")

        self.model = model.eval()
        self.device = next(model.parameters()).device
        print(f"✓ Model device: {self.device}")

        train_dataset = datamodule.train_dataset
        self.feature_names = list(train_dataset.features_df.columns)
        self.feature_indices = train_dataset.feature_indices

        # Validate indices
        num_features = len(self.feature_names)
        for group_name, indices in self.feature_indices.items():
            invalid_indices = [idx for idx in indices if idx >= num_features]
            if invalid_indices:
                print(f"⚠ WARNING: Group '{group_name}' contains invalid indices: {invalid_indices}")
                print(f"   Total features: {num_features}, Max valid index: {num_features - 1}")

        # self.class_names = datamodule.label_encoder.classes_
        self.class_names = [str(cls) for cls in datamodule.label_encoder.classes_]

        # Store feature groups for grouped analysis
        self.feature_groups = datamodule.feature_groups

        try:
            self.protocol_col_index = self.feature_names.index('Protocol')
        except ValueError:
            print("⚠ Warning: 'Protocol' feature not found. Using index 0 as default.")
            self.protocol_col_index = 0

        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Classes: {len(self.class_names)}")
        print(f"✓ Attack types: {', '.join(self.class_names)}")

        print(f"\\n📊 Creating background distribution from {background_samples} samples...")
        background_df = train_dataset.features_df.sample(
            n=min(background_samples, len(train_dataset.features_df)),
            random_state=42
        )

        self.background_summary = shap.kmeans(background_df, 10)
        print("✓ Background summary created using K-means clustering")

        self.explainer = shap.KernelExplainer(self.predict_wrapper, self.background_summary)
        print("✓ SHAP KernelExplainer initialized\\n")

    def predict_wrapper(self, implicit_features_batch: np.ndarray) -> np.ndarray:
        """Wrapper to adapt raw features for model input."""
        input_tensor = torch.tensor(implicit_features_batch, dtype=torch.float32).to(self.device)
        batch_size = input_tensor.shape[0]
        protocol_ids = input_tensor[:, self.protocol_col_index].long().clamp(0, 17)

        explicit_features_dict = {}
        for group_name, indices in self.feature_indices.items():
            if len(indices) > 0:
                group_features = input_tensor[:, indices]
            else:
                group_features = torch.zeros(batch_size, 1, dtype=torch.float32).to(self.device)
            explicit_features_dict[group_name] = group_features

        with torch.no_grad():
            logits = self.model(input_tensor, explicit_features_dict, protocol_ids)
            probs = torch.nn.functional.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def _create_feature_importance_plot(self, shap_values: np.ndarray,
                                        feature_names: List[str],
                                        save_path: Path,
                                        top_k: int = 15,
                                        title: str = "Feature Importance"):
        """Create a horizontal bar plot showing feature importance."""
        abs_shap = np.abs(shap_values)
        sorted_idx = np.argsort(abs_shap)[-top_k:]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [COLORS['accent'] if shap_values[i] > 0 else COLORS['secondary']
                  for i in sorted_idx]

        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, shap_values[sorted_idx], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['accent'], label='Increases prediction'),
            Patch(facecolor=COLORS['secondary'], label='Decreases prediction')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _create_feature_group_analysis(self, shap_values: np.ndarray,
                                       sample_values: np.ndarray,
                                       save_path: Path):
        """Analyze and visualize feature importance by groups (temporal, spatial, etc.)."""
        # DEBUG: Print dimensions
        print(f"DEBUG: shap_values.shape = {shap_values.shape}")
        print(f"DEBUG: sample_values.shape = {sample_values.shape}")

        group_impacts = {}

        # Add bounds checking
        max_valid_index = shap_values.shape[0] - 1

        for group_name, indices in self.feature_indices.items():
            if len(indices) > 0:
                # Filter indices to only include valid ones
                valid_indices = [idx for idx in indices if idx <= max_valid_index]

                if len(valid_indices) == 0:
                    continue

                group_shap = shap_values[valid_indices]
                group_impacts[group_name] = {
                    'total_impact': np.abs(group_shap).sum(),
                    'mean_impact': np.abs(group_shap).mean(),
                    'positive_impact': group_shap[group_shap > 0].sum(),
                    'negative_impact': group_shap[group_shap < 0].sum(),
                    'num_features': len(valid_indices)
                }

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Total impact by group
        groups = list(group_impacts.keys())
        total_impacts = [group_impacts[g]['total_impact'] for g in groups]
        colors_palette = [COLORS['primary'], COLORS['accent'], COLORS['success'], COLORS['secondary']]

        ax1.bar(groups, total_impacts, color=colors_palette[:len(groups)], alpha=0.8)
        ax1.set_ylabel('Total Absolute SHAP Value', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Group Impact', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Positive vs Negative impact
        pos_impacts = [group_impacts[g]['positive_impact'] for g in groups]
        neg_impacts = [np.abs(group_impacts[g]['negative_impact']) for g in groups]

        x = np.arange(len(groups))
        width = 0.35

        ax2.bar(x - width / 2, pos_impacts, width, label='Positive Impact',
                color=COLORS['success'], alpha=0.8)
        ax2.bar(x + width / 2, neg_impacts, width, label='Negative Impact',
                color=COLORS['danger'], alpha=0.8)
        ax2.set_ylabel('SHAP Value Magnitude', fontsize=11, fontweight='bold')
        ax2.set_title('Directional Impact by Group', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups, rotation=45)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return group_impacts

    def _create_decision_plot(self, shap_values: np.ndarray,
                              expected_value: float,
                              sample_values: np.ndarray,
                              pred_class_name: str,
                              save_path: Path):
        """Create a decision plot showing how features contribute to final prediction."""
        try:
            plt.figure(figsize=(10, 8))
            shap.decision_plot(
                expected_value,
                shap_values,
                feature_names=self.feature_names,
                show=False,
                highlight=0
            )
            plt.title(f'Decision Path for {pred_class_name} Prediction',
                      fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            print(f"⚠ Could not create decision plot: {e}")
            return False

    def _generate_text_report(self, sample_idx: int,
                              pred_class_name: str,
                              pred_prob: float,
                              true_label: Optional[str],
                              top_features: List[Tuple[str, float]],
                              group_impacts: Dict,
                              save_path: Path):
        """Generate comprehensive text report."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\\n")
            f.write("SHAP EXPLAINABILITY REPORT\\n")
            f.write("=" * 80 + "\\n\\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Sample Index: {sample_idx}\\n\\n")

            f.write("-" * 80 + "\\n")
            f.write("PREDICTION SUMMARY\\n")
            f.write("-" * 80 + "\\n\\n")

            if true_label:
                f.write(f"True Label: {true_label}\\n")
                f.write(f"Predicted Label: {pred_class_name}\\n")
                f.write(f"Confidence: {pred_prob:.2%}\\n")
                f.write(f"Correct: {'✓ YES' if true_label == pred_class_name else '✗ NO'}\\n\\n")
            else:
                f.write(f"Predicted Label: {pred_class_name}\\n")
                f.write(f"Confidence: {pred_prob:.2%}\\n\\n")

            f.write("-" * 80 + "\\n")
            f.write("TOP INFLUENTIAL FEATURES\\n")
            f.write("-" * 80 + "\\n\\n")

            for rank, (feat_name, shap_val) in enumerate(top_features, 1):
                direction = "↑ INCREASES" if shap_val > 0 else "↓ DECREASES"
                f.write(f"{rank:2d}. {feat_name:35s} {direction} prediction by {abs(shap_val):.4f}\\n")

            f.write("\\n" + "-" * 80 + "\\n")
            f.write("FEATURE GROUP ANALYSIS\\n")
            f.write("-" * 80 + "\\n\\n")

            for group_name, stats in group_impacts.items():
                f.write(f"{group_name.upper()} GROUP:\\n")
                f.write(f"  • Total Impact: {stats['total_impact']:.4f}\\n")
                f.write(f"  • Average Impact: {stats['mean_impact']:.4f}\\n")
                f.write(f"  • Positive Impact: {stats['positive_impact']:.4f}\\n")
                f.write(f"  • Negative Impact: {stats['negative_impact']:.4f}\\n")
                f.write(f"  • Features: {stats['num_features']}\\n\\n")

            f.write("-" * 80 + "\\n")
            f.write("INTERPRETATION GUIDE\\n")
            f.write("-" * 80 + "\\n\\n")
            f.write("• SHAP values represent each feature's contribution to the prediction\\n")
            f.write("• Positive values push the prediction toward the predicted class\\n")
            f.write("• Negative values push the prediction away from the predicted class\\n")
            f.write("• Larger absolute values indicate stronger influence\\n\\n")

            # Add insights based on prediction
            f.write("-" * 80 + "\\n")
            f.write("KEY INSIGHTS\\n")
            f.write("-" * 80 + "\\n\\n")

            # Identify dominant feature group
            dominant_group = max(group_impacts.items(), key=lambda x: x[1]['total_impact'])
            f.write(f"• The {dominant_group[0].upper()} feature group has the strongest\\n")
            f.write(f"  influence on this prediction ({dominant_group[1]['total_impact']:.4f} total impact)\\n\\n")

            # Identify most influential feature
            top_feature = top_features[0]
            f.write(f"• The most influential feature is '{top_feature[0]}'\\n")
            f.write(f"  with a SHAP value of {top_feature[1]:.4f}\\n\\n")

    def explain_prediction(self, sample_df_row: pd.Series,
                           sample_idx: int = 0,
                           true_label: Optional[str] = None,
                           output_dir: str = ".",
                           show_plot: bool = False,
                           save_html: bool = True):
        """
        Generate comprehensive explanation for a single prediction.

        Args:
            sample_df_row: Feature values for the sample
            sample_idx: Sample identifier
            true_label: Ground truth label (if available)
            output_dir: Directory to save reports
            show_plot: Whether to display plots interactively
            save_html: Whether to save interactive HTML plots
        """
        print(f"\\n{'=' * 80}")
        print(f"EXPLAINING SAMPLE {sample_idx}")
        print(f"{'=' * 80}\\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        sample_np = sample_df_row.values.reshape(1, -1)

        # Get prediction
        probs = self.predict_wrapper(sample_np)[0]
        pred_class_idx = np.argmax(probs)
        pred_class_name = self.class_names[pred_class_idx]
        pred_prob = probs[pred_class_idx]

        print(f"Predicted: {pred_class_name} ({pred_prob:.2%} confidence)")
        if true_label:
            print(f"True Label: {true_label}")
            print(f"Result: {'✓ CORRECT' if true_label == pred_class_name else '✗ INCORRECT'}\\n")

        print("🔍 Computing SHAP values (this may take 30-60 seconds)...")
        shap_values = self.explainer.shap_values(sample_np, nsamples=100, l1_reg=1e-10)

        # DEBUG: Check the shape
        print(f"DEBUG: Initial shap_values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"DEBUG: shap_values is a list with {len(shap_values)} elements")
            if len(shap_values) > 0:
                print(f"DEBUG: First element shape: {shap_values[0].shape}")
        else:
            print(f"DEBUG: shap_values shape: {shap_values.shape}")

        # FIXED: Handle SHAP output correctly for multi-class
        if isinstance(shap_values, list):
            # List of arrays, one per class: each array is (n_samples, n_features)
            shap_values_for_pred = shap_values[pred_class_idx][0]
            expected_value_for_pred = self.explainer.expected_value[pred_class_idx]
        else:
            # Single array: (n_samples, n_features) or (n_samples, n_features, n_classes)
            if len(shap_values.shape) == 3:
                # Shape is (n_samples, n_features, n_classes)
                shap_values_for_pred = shap_values[0, :, pred_class_idx]
            else:
                # Shape is (n_samples, n_features)
                shap_values_for_pred = shap_values[0]

            if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1:
                expected_value_for_pred = self.explainer.expected_value[pred_class_idx]
            else:
                expected_value_for_pred = self.explainer.expected_value

        # Verify the shape
        print(f"DEBUG: Final shap_values_for_pred.shape = {shap_values_for_pred.shape}")
        assert (
                shap_values_for_pred.shape[0] == len(self.feature_names)
        ), f"SHAP values shape {shap_values_for_pred.shape} doesn't match features {len(self.feature_names)}"

        # Get top features
        abs_shap = np.abs(shap_values_for_pred)
        sorted_indices = np.argsort(abs_shap)[::-1]
        top_features = [(self.feature_names[i], shap_values_for_pred[i])
                        for i in sorted_indices[:15]]

        print("✓ SHAP computation complete\\n")
        print("📊 Generating visualizations...\\n")

        # 1. Feature Importance Plot
        importance_path = output_path / "1_feature_importance.png"
        self._create_feature_importance_plot(
            shap_values_for_pred,
            self.feature_names,
            importance_path,
            top_k=15,
            title=f"Top Features for {pred_class_name} Prediction"
        )
        print(f"✓ Saved: {importance_path.name}")

        # 2. Feature Group Analysis
        group_analysis_path = output_path / "2_group_analysis.png"
        group_impacts = self._create_feature_group_analysis(
            shap_values_for_pred,
            sample_np[0],
            group_analysis_path
        )
        print(f"✓ Saved: {group_analysis_path.name}")

        # 3. Waterfall Plot
        waterfall_path = output_path / "3_waterfall_plot.png"
        try:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_for_pred,
                    base_values=expected_value_for_pred,
                    data=sample_np[0],
                    feature_names=self.feature_names
                ),
                max_display=15,
                show=False
            )
            plt.title(f'Feature Contributions to {pred_class_name} Prediction',
                      fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {waterfall_path.name}")
        except Exception as e:
            print(f"⚠ Could not create waterfall plot: {e}")

        # 4. Decision Plot
        decision_path = output_path / "4_decision_plot.png"
        if self._create_decision_plot(
                shap_values_for_pred,
                expected_value_for_pred,
                sample_np[0],
                pred_class_name,
                decision_path
        ):
            print(f"✓ Saved: {decision_path.name}")

        # 5. Force Plot (Interactive HTML)
        if save_html:
            force_plot_path = output_path / "5_force_plot.html"
            try:
                force_plot = shap.force_plot(
                    expected_value_for_pred,
                    shap_values_for_pred,
                    sample_np[0],
                    feature_names=self.feature_names,
                    matplotlib=False
                )
                shap.save_html(str(force_plot_path), force_plot)
                print(f"✓ Saved: {force_plot_path.name} (interactive)")
            except Exception as e:
                print(f"⚠ Could not create force plot: {e}")

        # 6. Text Report
        report_path = output_path / "EXPLANATION_REPORT.txt"
        self._generate_text_report(
            sample_idx,
            pred_class_name,
            pred_prob,
            true_label,
            top_features,
            group_impacts,
            report_path
        )
        print(f"✓ Saved: {report_path.name}")

        print(f"\\n{'=' * 80}")
        print(f"EXPLANATION COMPLETE - All files saved to: {output_path}")
        print(f"{'=' * 80}\\n")

        return {
            "shap_values": shap_values,
            "predicted_class_name": pred_class_name,
            "predicted_class_index": pred_class_idx,
            "confidence": pred_prob,
            "top_features": top_features,
            "group_impacts": group_impacts
        }

    def explain_batch(self, samples_df: pd.DataFrame,
                      output_dir: str = ".",
                      class_labels: Optional[List[str]] = None):
        """
        Generate comprehensive explanation for a batch of samples.

        Args:
            samples_df: DataFrame containing multiple samples
            output_dir: Directory to save reports
            class_labels: Optional ground truth labels for samples
        """
        print(f"\\n{'=' * 80}")
        print(f"BATCH EXPLANATION: {len(samples_df)} SAMPLES")
        print(f"{'=' * 80}\\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples_np = samples_df.values

        print("🔍 Computing SHAP values for batch (this may take several minutes)...")
        shap_values = self.explainer.shap_values(samples_np, nsamples=100, l1_reg=1e-10)
        print("✓ SHAP computation complete\\n")

        print("📊 Generating batch visualizations...\\n")

        # 1. Summary Bar Plot
        summary_bar_path = output_path / "1_batch_summary_bar.png"
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                samples_df,
                feature_names=self.feature_names,
                class_names=self.class_names,
                plot_type="bar",
                max_display=20,
                show=False
            )
            plt.title('Feature Importance Across All Classes',
                      fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(summary_bar_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {summary_bar_path.name}")
        except Exception as e:
            print(f"⚠ Error generating summary bar plot: {e}")

        # 2. Summary Beeswarm Plot
        summary_dot_path = output_path / "2_batch_summary_beeswarm.png"
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values,
                samples_df,
                feature_names=self.feature_names,
                class_names=self.class_names,
                plot_type="dot",
                max_display=20,
                show=False
            )
            plt.title('Feature Impact Distribution (All Samples)',
                      fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(summary_dot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {summary_dot_path.name}")
        except Exception as e:
            print(f"⚠ Error generating summary beeswarm plot: {e}")

        # 3. Per-Class Analysis
        print("\\n📈 Performing per-class analysis...")
        class_analysis_path = output_path / "3_per_class_importance.png"
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for class_idx, class_name in enumerate(self.class_names):
            ax = axes[class_idx]

            # Get SHAP values for this class
            if isinstance(shap_values, list) and len(shap_values) == len(self.class_names):
                class_shap = shap_values[class_idx]
            else:
                if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                    # Shape (n_samples, n_features, n_classes)
                    class_shap = shap_values[:, :, class_idx]
                else:
                    # Fallback: use all SHAP values
                    class_shap = shap_values[0] if isinstance(shap_values, list) else shap_values

            # Average absolute SHAP values across samples
            mean_abs_shap = np.abs(class_shap).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[-10:]

            colors = [COLORS['primary']] * len(sorted_idx)
            ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=colors, alpha=0.8)
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([self.feature_names[i] for i in sorted_idx], fontsize=8)
            ax.set_xlabel('Mean |SHAP|', fontsize=9)
            ax.set_title(f'{class_name}', fontsize=10, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle('Top 10 Features Per Attack Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(class_analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {class_analysis_path.name}")

        # 4. Generate batch statistics report
        self._generate_batch_report(samples_df, shap_values, class_labels,
                                    output_path / "BATCH_REPORT.txt")

        print(f"\\n{'=' * 80}")
        print(f"BATCH EXPLANATION COMPLETE - All files saved to: {output_path}")
        print(f"{'=' * 80}\\n")

        return {
            "shap_values": shap_values,
            "output_dir": str(output_path)
        }

    def _generate_batch_report(self, samples_df: pd.DataFrame,
                               shap_values,
                               class_labels: Optional[List[str]],
                               save_path: Path):
        """Generate comprehensive batch analysis report."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\\n")
            f.write("BATCH SHAP ANALYSIS REPORT\\n")
            f.write("=" * 80 + "\\n\\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Number of Samples: {len(samples_df)}\\n")
            f.write(f"Number of Features: {len(self.feature_names)}\\n")
            f.write(f"Number of Classes: {len(self.class_names)}\\n\\n")

            f.write("-" * 80 + "\\n")
            f.write("OVERALL FEATURE IMPORTANCE\\n")
            f.write("-" * 80 + "\\n\\n")

            # FIXED: Handle SHAP values correctly
            if isinstance(shap_values, list):
                # List of arrays per class: concatenate all classes
                all_shap = np.concatenate([sv for sv in shap_values], axis=0)
            else:
                if len(shap_values.shape) == 3:
                    # Shape (n_samples, n_features, n_classes) - average over classes
                    all_shap = shap_values.mean(axis=2)
                else:
                    # Shape (n_samples, n_features)
                    all_shap = shap_values

            mean_abs_shap = np.abs(all_shap).mean(axis=0)

            # Verify shape
            print(f"DEBUG: mean_abs_shap.shape = {mean_abs_shap.shape}")
            assert (
                    mean_abs_shap.shape[0] == len(self.feature_names)
            ), f"SHAP shape mismatch: {mean_abs_shap.shape[0]} vs {len(self.feature_names)} features"

            sorted_idx = np.argsort(mean_abs_shap)[::-1]

            f.write("Top 20 Most Important Features (averaged across all samples):\\n\\n")
            for rank, idx in enumerate(sorted_idx[:20], 1):
                f.write(f"{rank:2d}. {self.feature_names[idx]:40s} {mean_abs_shap[idx]:.6f}\\n")

            f.write("\\n" + "-" * 80 + "\\n")
            f.write("FEATURE GROUP SUMMARY\\n")
            f.write("-" * 80 + "\\n\\n")

            for group_name, indices in self.feature_indices.items():
                if len(indices) > 0:
                    # Add bounds checking
                    valid_indices = [idx for idx in indices if idx < len(mean_abs_shap)]
                    if len(valid_indices) > 0:
                        group_importance = mean_abs_shap[valid_indices].mean()
                        f.write(f"{group_name.upper():20s} - Mean Importance: {group_importance:.6f}\\n")

    def get_top_features(self, sample_df_row: pd.Series, top_k: int = 10):
        """
        Get top K most influential features for a single sample.

        Returns:
            Tuple of (top_features, predicted_class_name)
        """
        print(f"\\n🔍 Analyzing top {top_k} features...")

        sample_np = sample_df_row.values.reshape(1, -1)
        probs = self.predict_wrapper(sample_np)[0]
        pred_class_idx = np.argmax(probs)
        pred_class_name = self.class_names[pred_class_idx]

        shap_values = self.explainer.shap_values(sample_np, nsamples=100, l1_reg=1e-10)

        # FIXED: Handle SHAP output correctly
        if isinstance(shap_values, list):
            class_shap_values = shap_values[pred_class_idx][0]
        else:
            if len(shap_values.shape) == 3:
                class_shap_values = shap_values[0, :, pred_class_idx]
            else:
                class_shap_values = shap_values[0]

        abs_shap = np.abs(class_shap_values)
        sorted_indices = np.argsort(abs_shap)[::-1]

        top_features = []
        for i in sorted_indices[:top_k]:
            feature_name = self.feature_names[i]
            shap_value = class_shap_values[i]
            feature_value = sample_np[0, i]

            top_features.append({
                'feature': feature_name,
                'shap_value': shap_value,
                'feature_value': feature_value,
                'impact': 'POSITIVE' if shap_value > 0 else 'NEGATIVE'
            })

        print(f"\\n✓ Prediction: {pred_class_name} ({probs[pred_class_idx]:.2%} confidence)")
        print(f"\\nTop {top_k} Influential Features:\\n")
        print(f"{'Rank':<6}{'Feature':<35}{'SHAP Value':<15}{'Impact':<10}")
        print("-" * 70)

        for rank, feat_info in enumerate(top_features, 1):
            print(f"{rank:<6}{feat_info['feature']:<35}{feat_info['shap_value']:>+.4f}     "
                  f"{feat_info['impact']:<10}")

        return top_features, pred_class_name

    def compare_samples(self, sample_indices: List[int],
                        test_dataset,
                        output_dir: str = "."):
        """
        Compare explanations across multiple samples side-by-side.
        Useful for understanding model behavior on different attack types.
        """
        print(f"\\n{'=' * 80}")
        print(f"COMPARING {len(sample_indices)} SAMPLES")
        print(f"{'=' * 80}\\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples_data = []
        for idx in sample_indices:
            sample_row = test_dataset.features_df.iloc[idx]
            label = test_dataset.labels[idx]
            label_name = self.class_names[label]

            samples_data.append({
                'index': idx,
                'features': sample_row,
                'label': label_name
            })

        # Create comparison visualization
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 4 * len(sample_indices)))
        if len(sample_indices) == 1:
            axes = [axes]

        for i, (sample, ax) in enumerate(zip(samples_data, axes)):
            sample_np = sample['features'].values.reshape(1, -1)
            shap_vals = self.explainer.shap_values(sample_np, nsamples=100, l1_reg=1e-10)

            # FIXED: Handle SHAP output correctly
            if isinstance(shap_vals, list):
                pred_idx = np.argmax(self.predict_wrapper(sample_np)[0])
                shap_for_plot = shap_vals[pred_idx][0]
            else:
                if len(shap_vals.shape) == 3:
                    pred_idx = np.argmax(self.predict_wrapper(sample_np)[0])
                    shap_for_plot = shap_vals[0, :, pred_idx]
                else:
                    shap_for_plot = shap_vals[0]

            # Plot top 10 features
            abs_shap = np.abs(shap_for_plot)
            sorted_idx = np.argsort(abs_shap)[-10:]
            colors = [COLORS['accent'] if shap_for_plot[i] > 0 else COLORS['secondary']
                      for i in sorted_idx]

            ax.barh(range(len(sorted_idx)), shap_for_plot[sorted_idx], color=colors, alpha=0.8)
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([self.feature_names[i] for i in sorted_idx], fontsize=9)
            ax.set_xlabel('SHAP Value', fontsize=10)
            ax.set_title(f'Sample {sample["index"]} - True Label: {sample["label"]}',
                         fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle('Cross-Sample Feature Comparison', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        comparison_path = output_path / "sample_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Comparison saved to: {comparison_path}")
        print(f"\\n{'=' * 80}\\n")