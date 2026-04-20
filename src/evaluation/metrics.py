"""
Evaluation metrics and assessment tools for static code analysis.

This module provides comprehensive evaluation metrics for vulnerability
detection models and bug triage systems.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class VulnerabilityEvaluator:
    """Comprehensive evaluator for vulnerability detection models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the evaluator."""
        self.config = config or {}
        self.results = {}
    
    def evaluate_detection_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        vulnerability_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate vulnerability detection performance."""
        logger.info("Evaluating vulnerability detection performance")
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        unique_labels = np.unique(y_true)
        per_class_metrics = {}
        
        for label in unique_labels:
            label_name = vulnerability_types[label] if vulnerability_types and label < len(vulnerability_types) else f'class_{label}'
            per_class_metrics[label_name] = {
                'precision': precision_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0],
                'recall': recall_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0],
                'f1': f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
            }
        
        metrics['per_class'] = per_class_metrics
        
        # ROC and PR curves if probabilities available
        if y_proba is not None:
            if len(unique_labels) == 2:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
                
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba[:, 1])
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba[:, 1])
                metrics['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
            else:
                # Multi-class classification
                metrics['auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=vulnerability_types,
            output_dict=True,
            zero_division=0
        )
        
        self.results['detection_performance'] = metrics
        return metrics
    
    def evaluate_precision_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate precision@K for bug triage."""
        logger.info("Evaluating precision@K metrics")
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        precision_at_k = {}
        
        for k in k_values:
            if k > len(sorted_labels):
                k = len(sorted_labels)
            
            # Count relevant items in top-k
            relevant_count = np.sum(sorted_labels[:k])
            precision_at_k[f'precision@{k}'] = relevant_count / k
        
        self.results['precision_at_k'] = precision_at_k
        return precision_at_k
    
    def evaluate_vulnerability_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        vulnerability_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate recall for different vulnerability types."""
        logger.info("Evaluating vulnerability recall by type")
        
        unique_labels = np.unique(y_true)
        recall_by_type = {}
        
        for label in unique_labels:
            label_name = vulnerability_types[label] if vulnerability_types and label < len(vulnerability_types) else f'type_{label}'
            
            # Count true positives and false negatives
            true_positives = np.sum((y_true == label) & (y_pred == label))
            false_negatives = np.sum((y_true == label) & (y_pred != label))
            
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            recall_by_type[label_name] = {
                'recall': recall,
                'true_positives': int(true_positives),
                'false_negatives': int(false_negatives),
                'total_samples': int(np.sum(y_true == label))
            }
        
        self.results['vulnerability_recall'] = recall_by_type
        return recall_by_type
    
    def evaluate_false_positive_rate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_tpr: float = 0.95
    ) -> Dict[str, float]:
        """Evaluate false positive rate at target true positive rate."""
        logger.info(f"Evaluating FPR at TPR={target_tpr}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # For binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            fpr_metrics = {
                'fpr': fpr,
                'tpr': tpr,
                'fpr_at_target_tpr': fpr if tpr >= target_tpr else None
            }
        else:
            # Multi-class: calculate average FPR
            fpr_values = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fpr_values.append(fpr)
            
            fpr_metrics = {
                'fpr_macro': np.mean(fpr_values),
                'fpr_std': np.std(fpr_values)
            }
        
        self.results['false_positive_rate'] = fpr_metrics
        return fpr_metrics
    
    def evaluate_alert_volume(
        self,
        y_pred: np.ndarray,
        vulnerability_types: Optional[List[str]] = None,
        severity_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate alert volume and workload metrics."""
        logger.info("Evaluating alert volume metrics")
        
        # Count alerts by type
        unique_labels, counts = np.unique(y_pred, return_counts=True)
        
        alert_counts = {}
        total_alerts = len(y_pred)
        
        for label, count in zip(unique_labels, counts):
            label_name = vulnerability_types[label] if vulnerability_types and label < len(vulnerability_types) else f'type_{label}'
            alert_counts[label_name] = {
                'count': int(count),
                'percentage': count / total_alerts * 100
            }
        
        # Calculate workload score if severity weights provided
        workload_score = 0
        if severity_weights:
            for label, count in zip(unique_labels, counts):
                label_name = vulnerability_types[label] if vulnerability_types and label < len(vulnerability_types) else f'type_{label}'
                weight = severity_weights.get(label_name, 1.0)
                workload_score += count * weight
        
        alert_metrics = {
            'total_alerts': total_alerts,
            'alert_counts': alert_counts,
            'workload_score': workload_score,
            'alerts_per_1000_lines': total_alerts / max(len(y_pred), 1) * 1000
        }
        
        self.results['alert_volume'] = alert_metrics
        return alert_metrics
    
    def evaluate_robustness(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, Any]:
        """Evaluate model robustness to adversarial perturbations."""
        logger.info("Evaluating model robustness")
        
        # Baseline accuracy
        baseline_pred = model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        
        robustness_results = {
            'baseline_accuracy': baseline_accuracy,
            'epsilon_results': {}
        }
        
        # Simple adversarial perturbation (add noise)
        for epsilon in epsilon_values:
            # Add random noise
            noise = np.random.normal(0, epsilon, X_test.shape)
            X_perturbed = X_test + noise
            
            # Make predictions
            try:
                perturbed_pred = model.predict(X_perturbed)
                perturbed_accuracy = accuracy_score(y_test, perturbed_pred)
                
                robustness_results['epsilon_results'][epsilon] = {
                    'accuracy': perturbed_accuracy,
                    'accuracy_drop': baseline_accuracy - perturbed_accuracy,
                    'robustness_ratio': perturbed_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Error evaluating robustness at epsilon={epsilon}: {e}")
                robustness_results['epsilon_results'][epsilon] = {
                    'accuracy': 0,
                    'accuracy_drop': baseline_accuracy,
                    'robustness_ratio': 0
                }
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def generate_leaderboard(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metric_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Generate a leaderboard comparing different models."""
        logger.info("Generating model leaderboard")
        
        if not model_results:
            return pd.DataFrame()
        
        # Default metric weights
        if metric_weights is None:
            metric_weights = {
                'accuracy': 0.3,
                'f1_macro': 0.3,
                'precision@5': 0.2,
                'auc': 0.2
            }
        
        leaderboard_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # Extract key metrics
            if 'detection_performance' in results:
                perf = results['detection_performance']
                row.update({
                    'accuracy': perf.get('accuracy', 0),
                    'f1_macro': perf.get('f1_macro', 0),
                    'precision_macro': perf.get('precision_macro', 0),
                    'recall_macro': perf.get('recall_macro', 0),
                    'auc': perf.get('auc', perf.get('auc_macro', 0))
                })
            
            if 'precision_at_k' in results:
                prec_k = results['precision_at_k']
                row.update({
                    'precision@1': prec_k.get('precision@1', 0),
                    'precision@5': prec_k.get('precision@5', 0),
                    'precision@10': prec_k.get('precision@10', 0)
                })
            
            # Calculate weighted score
            weighted_score = 0
            for metric, weight in metric_weights.items():
                if metric in row:
                    weighted_score += row[metric] * weight
            
            row['weighted_score'] = weighted_score
            leaderboard_data.append(row)
        
        # Create DataFrame and sort by weighted score
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('weighted_score', ascending=False)
        
        self.results['leaderboard'] = leaderboard
        return leaderboard
    
    def plot_evaluation_results(self, save_path: Optional[str] = None) -> None:
        """Plot evaluation results."""
        logger.info("Plotting evaluation results")
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Static Code Analysis Evaluation Results', fontsize=16)
        
        # Plot 1: Confusion Matrix
        if 'detection_performance' in self.results:
            cm = np.array(self.results['detection_performance']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
        
        # Plot 2: ROC Curve
        if 'detection_performance' in self.results and 'roc_curve' in self.results['detection_performance']:
            roc_data = self.results['detection_performance']['roc_curve']
            axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'], label='ROC Curve')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # Plot 3: Precision-Recall Curve
        if 'detection_performance' in self.results and 'pr_curve' in self.results['detection_performance']:
            pr_data = self.results['detection_performance']['pr_curve']
            axes[1, 0].plot(pr_data['recall'], pr_data['precision'], label='PR Curve')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].legend()
        
        # Plot 4: Precision@K
        if 'precision_at_k' in self.results:
            prec_k = self.results['precision_at_k']
            k_values = [int(k.split('@')[1]) for k in prec_k.keys()]
            prec_values = list(prec_k.values())
            axes[1, 1].bar(k_values, prec_values)
            axes[1, 1].set_xlabel('K')
            axes[1, 1].set_ylabel('Precision@K')
            axes[1, 1].set_title('Precision@K')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a comprehensive summary report."""
        summary = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'total_metrics': len(self.results),
            'results': self.results
        }
        
        # Add key performance indicators
        if 'detection_performance' in self.results:
            perf = self.results['detection_performance']
            summary['key_metrics'] = {
                'accuracy': perf.get('accuracy', 0),
                'f1_macro': perf.get('f1_macro', 0),
                'auc': perf.get('auc', perf.get('auc_macro', 0))
            }
        
        return summary
