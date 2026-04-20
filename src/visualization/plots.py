"""
Visualization utilities for static code analysis results.

This module provides plotting and visualization functions
for analysis results, metrics, and model performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization creation and styling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the visualization manager."""
        self.config = config or {}
        self.style_config = self.config.get('visualization', {})
        
        # Set up matplotlib style
        self._setup_matplotlib()
    
    def _setup_matplotlib(self) -> None:
        """Set up matplotlib styling."""
        style = self.style_config.get('style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except OSError:
            logger.warning(f"Style {style} not available, using default")
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = self.style_config.get('figure_size', [12, 8])
        plt.rcParams['figure.dpi'] = self.style_config.get('dpi', 300)
        plt.rcParams['savefig.dpi'] = self.style_config.get('dpi', 300)
        plt.rcParams['savefig.bbox'] = 'tight'
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve."""
        plt.figure(figsize=(10, 8))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        average_precision: float,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision-recall curve."""
        plt.figure(figsize=(10, 8))
        
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {average_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        top_k: int = 15,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance."""
        # Get top K features
        sorted_indices = np.argsort(importance_scores)[::-1][:top_k]
        top_features = [feature_names[i] for i in sorted_indices]
        top_scores = [importance_scores[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        
        bars = plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_at_k(
        self,
        k_values: List[int],
        precision_values: List[float],
        title: str = "Precision@K",
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision@K curve."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(k_values, precision_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('K')
        plt.ylabel('Precision@K')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision@K plot saved to {save_path}")
        
        plt.show()
    
    def plot_vulnerability_distribution(
        self,
        vulnerability_counts: Dict[str, int],
        title: str = "Vulnerability Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """Plot vulnerability type distribution."""
        plt.figure(figsize=(12, 8))
        
        types = list(vulnerability_counts.keys())
        counts = list(vulnerability_counts.values())
        
        bars = plt.bar(types, counts)
        plt.xlabel('Vulnerability Type')
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        
        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Vulnerability distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_severity_distribution(
        self,
        severity_counts: Dict[str, int],
        title: str = "Severity Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """Plot severity distribution pie chart."""
        plt.figure(figsize=(10, 8))
        
        labels = list(severity_counts.keys())
        sizes = list(severity_counts.values())
        
        # Define colors for severity levels
        severity_colors = {
            'critical': '#d62728',
            'high': '#ff7f0e',
            'medium': '#ffbb78',
            'low': '#2ca02c',
            'info': '#17a2b8'
        }
        
        colors = [severity_colors.get(label, '#999999') for label in labels]
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Severity distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_robustness_analysis(
        self,
        epsilon_values: List[float],
        accuracies: List[float],
        baseline_accuracy: float,
        title: str = "Model Robustness",
        save_path: Optional[str] = None
    ) -> None:
        """Plot robustness analysis."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(epsilon_values, accuracies, marker='o', linewidth=2, markersize=8, label='Perturbed Accuracy')
        plt.axhline(y=baseline_accuracy, color='r', linestyle='--', 
                   label=f'Baseline Accuracy: {baseline_accuracy:.3f}')
        
        plt.xlabel('Epsilon (Perturbation Strength)')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Robustness analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison bar chart."""
        plt.figure(figsize=(12, 8))
        
        models = list(model_results.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [model_results[model].get(metric, 0) for metric in metrics]
            plt.bar(x + i * width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(x + width * (len(models) - 1) / 2, metrics)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(
        self,
        vulnerabilities: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive dashboard using Plotly."""
        if not vulnerabilities:
            logger.warning("No vulnerabilities to visualize")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(vulnerabilities)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Severity Distribution', 'Vulnerability Types', 
                          'Confidence Distribution', 'Line Number Distribution'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Severity distribution
        severity_counts = df['severity'].value_counts()
        fig.add_trace(
            go.Pie(labels=severity_counts.index, values=severity_counts.values, name="Severity"),
            row=1, col=1
        )
        
        # Vulnerability types
        type_counts = df['vulnerability_type'].value_counts()
        fig.add_trace(
            go.Bar(x=type_counts.index, y=type_counts.values, name="Types"),
            row=1, col=2
        )
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(x=df['confidence'], name="Confidence"),
            row=2, col=1
        )
        
        # Line number distribution
        fig.add_trace(
            go.Histogram(x=df['line_number'], name="Line Numbers"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Vulnerability Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        title: str = "Learning Curves",
        save_path: Optional[str] = None
    ) -> None:
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, 'b-', label='Training Score')
        plt.plot(epochs, val_scores, 'r-', label='Validation Score')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def save_all_plots(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Save all plots from results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving plots to {output_path}")
        
        # Confusion matrix
        if 'detection_performance' in results and 'confusion_matrix' in results['detection_performance']:
            cm = np.array(results['detection_performance']['confusion_matrix'])
            self.plot_confusion_matrix(
                cm,
                save_path=str(output_path / 'confusion_matrix.png')
            )
        
        # Feature importance
        if 'feature_importance' in results:
            feature_importance = results['feature_importance']
            if feature_importance:
                features = list(feature_importance.keys())
                scores = list(feature_importance.values())
                self.plot_feature_importance(
                    features, scores,
                    save_path=str(output_path / 'feature_importance.png')
                )
        
        # Precision@K
        if 'precision_at_k' in results:
            prec_k = results['precision_at_k']
            k_values = [int(k.split('@')[1]) for k in prec_k.keys()]
            prec_values = list(prec_k.values())
            self.plot_precision_at_k(
                k_values, prec_values,
                save_path=str(output_path / 'precision_at_k.png')
            )
        
        # Robustness
        if 'robustness' in results:
            robustness = results['robustness']
            if 'epsilon_results' in robustness:
                epsilons = list(robustness['epsilon_results'].keys())
                accuracies = [robustness['epsilon_results'][eps]['accuracy'] for eps in epsilons]
                baseline = robustness['baseline_accuracy']
                self.plot_robustness_analysis(
                    epsilons, accuracies, baseline,
                    save_path=str(output_path / 'robustness.png')
                )
        
        logger.info("All plots saved successfully")


def create_summary_report(
    results: Dict[str, Any],
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Create a comprehensive summary report with visualizations."""
    logger.info("Creating summary report")
    
    viz_manager = VisualizationManager(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all plots
    viz_manager.save_all_plots(results, str(output_path))
    
    # Create summary text file
    summary_path = output_path / 'summary_report.txt'
    
    with open(summary_path, 'w') as f:
        f.write("Static Code Analysis - Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Detection performance
        if 'detection_performance' in results:
            perf = results['detection_performance']
            f.write("Detection Performance:\n")
            f.write(f"  Accuracy: {perf.get('accuracy', 0):.3f}\n")
            f.write(f"  Precision (Macro): {perf.get('precision_macro', 0):.3f}\n")
            f.write(f"  Recall (Macro): {perf.get('recall_macro', 0):.3f}\n")
            f.write(f"  F1 Score (Macro): {perf.get('f1_macro', 0):.3f}\n")
            f.write(f"  AUC: {perf.get('auc', perf.get('auc_macro', 0)):.3f}\n\n")
        
        # Precision@K
        if 'precision_at_k' in results:
            prec_k = results['precision_at_k']
            f.write("Precision@K:\n")
            for k, v in prec_k.items():
                f.write(f"  {k}: {v:.3f}\n")
            f.write("\n")
        
        # Feature importance
        if 'feature_importance' in results:
            f.write("Top 10 Features:\n")
            sorted_features = sorted(
                results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for feature, importance in sorted_features:
                f.write(f"  {feature}: {importance:.3f}\n")
            f.write("\n")
    
    logger.info(f"Summary report saved to {summary_path}")
