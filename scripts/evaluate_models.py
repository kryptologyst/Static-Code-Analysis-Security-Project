#!/usr/bin/env python3
"""
Evaluation script for vulnerability detection models.

This script evaluates trained models on test datasets
and generates comprehensive performance reports.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from src.data.generator import CodeDatasetGenerator
from src.data.features import CodeFeatureExtractor
from src.models.vulnerability_detector import VulnerabilityDetector, VulnerabilityRanker
from src.evaluation.metrics import VulnerabilityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_models(model_dir: str) -> Dict[str, Any]:
    """Load trained models."""
    logger.info(f"Loading models from {model_dir}")
    
    models = {}
    model_path = Path(model_dir)
    
    # Load detector
    detector_path = model_path / 'vulnerability_detector.pkl'
    if detector_path.exists():
        detector = VulnerabilityDetector()
        detector.load_model(detector_path)
        models['detector'] = detector
        logger.info("Loaded vulnerability detector")
    else:
        logger.warning(f"Detector model not found: {detector_path}")
    
    # Load ranker
    ranker_path = model_path / 'vulnerability_ranker.pkl'
    if ranker_path.exists():
        ranker = VulnerabilityRanker()
        ranker.load_model(ranker_path)
        models['ranker'] = ranker
        logger.info("Loaded vulnerability ranker")
    else:
        logger.warning(f"Ranker model not found: {ranker_path}")
    
    return models


def generate_test_dataset(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Generate test dataset."""
    logger.info("Generating test dataset")
    
    dataset_config = config.get('dataset', {})
    generator = CodeDatasetGenerator(dataset_config)
    
    # Generate larger test dataset
    dataset = generator.generate_dataset(
        num_vulnerable=dataset_config.get('num_vulnerable_samples', 200) * 2,
        num_safe=dataset_config.get('num_safe_samples', 300) * 2,
        split_ratios=[0.5, 0.25, 0.25]  # More test data
    )
    
    # Convert to DataFrames
    dataset_dfs = {}
    for split_name, samples in dataset.items():
        dataset_dfs[split_name] = pd.DataFrame(samples)
    
    return dataset_dfs


def prepare_test_features(dataset: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Prepare features for testing."""
    logger.info("Preparing test features")
    
    feature_extractor = CodeFeatureExtractor(config.get('features', {}))
    
    # Prepare all data
    all_samples = []
    all_labels = []
    
    for split_name, df in dataset.items():
        all_samples.extend(df['code'].tolist())
        all_labels.extend(df['severity'].tolist())
    
    # Convert severity labels to numeric
    severity_mapping = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
    all_labels_numeric = [severity_mapping.get(label, 0) for label in all_labels]
    
    # Extract features
    X_all = feature_extractor.fit_transform(all_samples)
    y_all = np.array(all_labels_numeric)
    
    # Split into train/test for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': [f'feature_{i}' for i in range(X_all.shape[1])]
    }


def evaluate_models(models: Dict[str, Any], features: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all models."""
    logger.info("Evaluating models")
    
    evaluator = VulnerabilityEvaluator(config.get('evaluation', {}))
    results = {}
    
    # Evaluate detector
    if 'detector' in models:
        detector = models['detector']
        
        # Make predictions
        y_pred = detector.predict(features['X_test'])
        y_proba = detector.predict_proba(features['X_test'])
        
        # Evaluate detection performance
        detection_metrics = evaluator.evaluate_detection_performance(
            features['y_test'],
            y_pred,
            y_proba
        )
        
        # Evaluate precision@K
        precision_k_metrics = evaluator.evaluate_precision_at_k(
            features['y_test'],
            y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten(),
            config.get('evaluation', {}).get('precision_at_k_values', [1, 5, 10])
        )
        
        # Evaluate vulnerability recall
        vulnerability_types = ['safe', 'low', 'medium', 'high', 'critical']
        recall_metrics = evaluator.evaluate_vulnerability_recall(
            features['y_test'],
            y_pred,
            vulnerability_types
        )
        
        # Evaluate false positive rate
        fpr_metrics = evaluator.evaluate_false_positive_rate(
            features['y_test'],
            y_pred
        )
        
        # Evaluate alert volume
        alert_metrics = evaluator.evaluate_alert_volume(
            y_pred,
            vulnerability_types
        )
        
        # Evaluate robustness
        robustness_metrics = evaluator.evaluate_robustness(
            detector,
            features['X_test'],
            features['y_test']
        )
        
        results['detector'] = {
            'detection_performance': detection_metrics,
            'precision_at_k': precision_k_metrics,
            'vulnerability_recall': recall_metrics,
            'false_positive_rate': fpr_metrics,
            'alert_volume': alert_metrics,
            'robustness': robustness_metrics
        }
        
        # Feature importance
        feature_importance = detector.get_feature_importance()
        results['detector']['feature_importance'] = feature_importance
    
    # Evaluate ranker
    if 'ranker' in models:
        ranker = models['ranker']
        
        # Create mock vulnerability data for ranking
        test_vulnerabilities = []
        for i in range(len(features['X_test'])):
            vuln_data = {
                'code_snippet': f'test_code_{i}',
                'vulnerability_type': 'test_vulnerability',
                'severity': 'medium'
            }
            test_vulnerabilities.append(vuln_data)
        
        # Rank vulnerabilities
        ranked_vulnerabilities = ranker.rank_vulnerabilities(test_vulnerabilities)
        
        results['ranker'] = {
            'total_ranked': len(ranked_vulnerabilities),
            'avg_confidence': np.mean([v['priority_confidence'] for v in ranked_vulnerabilities])
        }
    
    return results


def generate_leaderboard(results: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Generate model leaderboard."""
    logger.info("Generating model leaderboard")
    
    evaluator = VulnerabilityEvaluator()
    leaderboard = evaluator.generate_leaderboard(results)
    
    return leaderboard


def create_visualizations(results: Dict[str, Any], output_dir: str) -> None:
    """Create visualization plots."""
    logger.info(f"Creating visualizations in {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Confusion Matrix
    if 'detector' in results and 'detection_performance' in results['detector']:
        cm = np.array(results['detector']['detection_performance']['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Vulnerability Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Feature Importance
    if 'detector' in results and 'feature_importance' in results['detector']:
        feature_importance = results['detector']['feature_importance']
        
        # Get top 15 features
        top_features = dict(list(feature_importance.items())[:15])
        
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        importance = list(top_features.values())
        
        plt.barh(features, importance)
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Precision@K
    if 'detector' in results and 'precision_at_k' in results['detector']:
        prec_k = results['detector']['precision_at_k']
        
        plt.figure(figsize=(10, 6))
        k_values = [int(k.split('@')[1]) for k in prec_k.keys()]
        prec_values = list(prec_k.values())
        
        plt.plot(k_values, prec_values, marker='o', linewidth=2, markersize=8)
        plt.title('Precision@K')
        plt.xlabel('K')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'precision_at_k.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Robustness Analysis
    if 'detector' in results and 'robustness' in results['detector']:
        robustness = results['detector']['robustness']
        
        if 'epsilon_results' in robustness:
            epsilons = list(robustness['epsilon_results'].keys())
            accuracies = [robustness['epsilon_results'][eps]['accuracy'] for eps in epsilons]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8)
            plt.axhline(y=robustness['baseline_accuracy'], color='r', linestyle='--', 
                       label=f"Baseline: {robustness['baseline_accuracy']:.3f}")
            plt.title('Model Robustness to Adversarial Perturbations')
            plt.xlabel('Epsilon (Perturbation Strength)')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'robustness.png', dpi=300, bbox_inches='tight')
            plt.close()


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save evaluation results."""
    logger.info(f"Saving results to {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_path = output_path / 'evaluation_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save leaderboard as CSV
    if 'leaderboard' in results:
        leaderboard_path = output_path / 'leaderboard.csv'
        results['leaderboard'].to_csv(leaderboard_path, index=False)
    
    logger.info("Results saved successfully")


def print_summary(results: Dict[str, Any]) -> None:
    """Print evaluation summary."""
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    if 'detector' in results:
        detector_results = results['detector']
        
        if 'detection_performance' in detector_results:
            perf = detector_results['detection_performance']
            logger.info(f"Accuracy: {perf.get('accuracy', 0):.3f}")
            logger.info(f"Precision (Macro): {perf.get('precision_macro', 0):.3f}")
            logger.info(f"Recall (Macro): {perf.get('recall_macro', 0):.3f}")
            logger.info(f"F1 Score (Macro): {perf.get('f1_macro', 0):.3f}")
            logger.info(f"AUC: {perf.get('auc', perf.get('auc_macro', 0)):.3f}")
        
        if 'precision_at_k' in detector_results:
            prec_k = detector_results['precision_at_k']
            logger.info(f"Precision@1: {prec_k.get('precision@1', 0):.3f}")
            logger.info(f"Precision@5: {prec_k.get('precision@5', 0):.3f}")
            logger.info(f"Precision@10: {prec_k.get('precision@10', 0):.3f}")
        
        if 'alert_volume' in detector_results:
            alert_vol = detector_results['alert_volume']
            logger.info(f"Total Alerts: {alert_vol.get('total_alerts', 0)}")
            logger.info(f"Alerts per 1000 lines: {alert_vol.get('alerts_per_1000_lines', 0):.1f}")
    
    if 'leaderboard' in results:
        logger.info("\nModel Leaderboard:")
        logger.info(results['leaderboard'].to_string(index=False))
    
    logger.info("=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate vulnerability detection models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for evaluation results')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    np.random.seed(config.get('random_seed', 42))
    
    try:
        # Load models
        models = load_models(args.model_dir)
        
        if not models:
            logger.error("No models found to evaluate")
            return
        
        # Generate test dataset
        dataset = generate_test_dataset(config)
        
        # Prepare test features
        features = prepare_test_features(dataset, config)
        
        # Evaluate models
        results = evaluate_models(models, features, config)
        
        # Generate leaderboard
        leaderboard = generate_leaderboard(results, config)
        results['leaderboard'] = leaderboard
        
        # Create visualizations
        if args.generate_plots:
            create_visualizations(results, args.output_dir)
        
        # Save results
        save_results(results, args.output_dir)
        
        # Print summary
        print_summary(results)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
