"""
Test suite for static code analysis project.

This module contains unit tests for all major components
of the static code analysis system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os

from src.analysis.scanner import CodeScanner, Vulnerability, ASTAnalyzer
from src.data.features import CodeFeatureExtractor, VulnerabilityDataset
from src.data.generator import CodeDatasetGenerator
from src.models.vulnerability_detector import VulnerabilityDetector, VulnerabilityRanker
from src.evaluation.metrics import VulnerabilityEvaluator
from src.utils.explainability import PIIProtector, RuleEvidenceExtractor, ExplainabilityManager


class TestCodeScanner:
    """Test cases for CodeScanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = CodeScanner()
        self.sample_code = """
def login(username, password):
    if password == 'admin123':  # Hardcoded password
        return True
    return False

def process_input(user_input):
    result = eval(user_input)  # Dangerous eval
    return result
"""
    
    def test_scan_code_basic(self):
        """Test basic code scanning functionality."""
        vulnerabilities = self.scanner.scan_code(self.sample_code)
        
        assert len(vulnerabilities) > 0
        assert all(isinstance(v, Vulnerability) for v in vulnerabilities)
        
        # Check for hardcoded password
        password_vulns = [v for v in vulnerabilities if v.vulnerability_type == 'hardcoded_password']
        assert len(password_vulns) > 0
        
        # Check for eval usage
        eval_vulns = [v for v in vulnerabilities if v.vulnerability_type == 'eval_usage']
        assert len(eval_vulns) > 0
    
    def test_scan_code_empty(self):
        """Test scanning empty code."""
        vulnerabilities = self.scanner.scan_code("")
        assert len(vulnerabilities) == 0
    
    def test_scan_code_syntax_error(self):
        """Test scanning code with syntax errors."""
        bad_code = "def broken_function(\n    return 'missing colon'"
        vulnerabilities = self.scanner.scan_code(bad_code)
        
        # Should handle syntax errors gracefully
        assert isinstance(vulnerabilities, list)
    
    def test_get_summary_stats(self):
        """Test summary statistics generation."""
        vulnerabilities = self.scanner.scan_code(self.sample_code)
        stats = self.scanner.get_summary_stats(vulnerabilities)
        
        assert 'total' in stats
        assert 'by_severity' in stats
        assert 'by_type' in stats
        assert stats['total'] > 0


class TestASTAnalyzer:
    """Test cases for ASTAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ASTAnalyzer()
    
    def test_analyze_ast_basic(self):
        """Test basic AST analysis."""
        code = "def test(): return 'hello'"
        vulnerabilities = self.analyzer.analyze_ast(code)
        
        assert isinstance(vulnerabilities, list)
    
    def test_pattern_analysis(self):
        """Test pattern-based analysis."""
        code = "password = 'secret123'"
        vulnerabilities = self.analyzer.analyze_ast(code)
        
        assert len(vulnerabilities) > 0
        assert any(v.vulnerability_type == 'hardcoded_password' for v in vulnerabilities)
    
    def test_ast_analysis_functions(self):
        """Test AST analysis of function calls."""
        code = "result = eval('2 + 2')"
        vulnerabilities = self.analyzer.analyze_ast(code)
        
        assert len(vulnerabilities) > 0
        assert any(v.vulnerability_type == 'eval_usage' for v in vulnerabilities)


class TestCodeFeatureExtractor:
    """Test cases for CodeFeatureExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CodeFeatureExtractor()
    
    def test_extract_ast_features(self):
        """Test AST feature extraction."""
        code = """
def test_function():
    if True:
        for i in range(10):
            print(i)
"""
        features = self.extractor.extract_ast_features(code)
        
        assert isinstance(features, dict)
        assert 'ast_nodes' in features
        assert 'function_count' in features
        assert 'if_count' in features
        assert 'for_count' in features
        assert features['function_count'] == 1
        assert features['if_count'] == 1
        assert features['for_count'] == 1
    
    def test_extract_text_features(self):
        """Test text feature extraction."""
        code = """
# This is a comment
def hello():
    '''This is a docstring'''
    return "Hello World"
"""
        features = self.extractor.extract_text_features(code)
        
        assert isinstance(features, dict)
        assert 'line_count' in features
        assert 'char_count' in features
        assert 'comment_lines' in features
        assert 'docstring_lines' in features
        assert features['comment_lines'] >= 1
    
    def test_extract_all_features(self):
        """Test extraction of all features."""
        code = "def test(): return 42"
        features = self.extractor.extract_all_features(code)
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        code_samples = [
            "def test1(): return 1",
            "def test2(): return 2",
            "def test3(): return 3"
        ]
        
        features = self.extractor.fit_transform(code_samples)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3
        assert features.shape[1] > 0


class TestVulnerabilityDetector:
    """Test cases for VulnerabilityDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = VulnerabilityDetector(model_type='logistic_regression')
        
        # Create sample data
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 3, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 3, 20)
    
    def test_fit(self):
        """Test model fitting."""
        self.detector.fit(self.X_train, self.y_train)
        assert self.detector.is_fitted
    
    def test_predict(self):
        """Test prediction functionality."""
        self.detector.fit(self.X_train, self.y_train)
        predictions = self.detector.predict(self.X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.X_test)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        self.detector.fit(self.X_train, self.y_train)
        probabilities = self.detector.predict_proba(self.X_test)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(self.X_test)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        self.detector.fit(self.X_train, self.y_train)
        importance = self.detector.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.detector.fit(self.X_train, self.y_train)
        metrics = self.detector.evaluate(self.X_test, self.y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics


class TestCodeDatasetGenerator:
    """Test cases for CodeDatasetGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CodeDatasetGenerator({'random_seed': 42})
    
    def test_generate_vulnerable_code(self):
        """Test vulnerable code generation."""
        samples = self.generator.generate_vulnerable_code(10)
        
        assert len(samples) == 10
        assert all('code' in sample for sample in samples)
        assert all('vulnerability_type' in sample for sample in samples)
        assert all('severity' in sample for sample in samples)
    
    def test_generate_safe_code(self):
        """Test safe code generation."""
        samples = self.generator.generate_safe_code(10)
        
        assert len(samples) == 10
        assert all('code' in sample for sample in samples)
        assert all(sample['vulnerability_type'] == 'safe' for sample in samples)
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        dataset = self.generator.generate_dataset(50, 50)
        
        assert 'train' in dataset
        assert 'validation' in dataset
        assert 'test' in dataset
        
        assert len(dataset['train']) > 0
        assert len(dataset['validation']) > 0
        assert len(dataset['test']) > 0
    
    def test_get_dataset_stats(self):
        """Test dataset statistics generation."""
        dataset = self.generator.generate_dataset(20, 20)
        stats = self.generator.get_dataset_stats(dataset)
        
        assert isinstance(stats, dict)
        assert 'train' in stats
        assert 'validation' in stats
        assert 'test' in stats


class TestVulnerabilityEvaluator:
    """Test cases for VulnerabilityEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = VulnerabilityEvaluator()
        
        # Create sample data
        self.y_true = np.array([0, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0, 1])
        self.y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    
    def test_evaluate_detection_performance(self):
        """Test detection performance evaluation."""
        metrics = self.evaluator.evaluate_detection_performance(
            self.y_true, self.y_pred, self.y_proba
        )
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
    
    def test_evaluate_precision_at_k(self):
        """Test precision@K evaluation."""
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        metrics = self.evaluator.evaluate_precision_at_k(self.y_true, y_scores)
        
        assert isinstance(metrics, dict)
        assert 'precision@1' in metrics
        assert 'precision@5' in metrics
    
    def test_evaluate_vulnerability_recall(self):
        """Test vulnerability recall evaluation."""
        metrics = self.evaluator.evaluate_vulnerability_recall(
            self.y_true, self.y_pred, ['safe', 'vulnerable']
        )
        
        assert isinstance(metrics, dict)
        assert 'safe' in metrics
        assert 'vulnerable' in metrics
    
    def test_evaluate_false_positive_rate(self):
        """Test false positive rate evaluation."""
        metrics = self.evaluator.evaluate_false_positive_rate(self.y_true, self.y_pred)
        
        assert isinstance(metrics, dict)
        assert 'fpr' in metrics or 'fpr_macro' in metrics


class TestPIIProtector:
    """Test cases for PIIProtector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.protector = PIIProtector()
    
    def test_detect_pii_email(self):
        """Test email detection."""
        text = "Contact us at test@example.com for more info"
        detected = self.protector.detect_pii(text)
        
        assert 'email' in detected
        assert 'test@example.com' in detected['email']
    
    def test_detect_pii_phone(self):
        """Test phone number detection."""
        text = "Call us at (555) 123-4567"
        detected = self.protector.detect_pii(text)
        
        assert 'phone' in detected
    
    def test_redact_pii(self):
        """Test PII redaction."""
        text = "Email: test@example.com, Phone: (555) 123-4567"
        redacted = self.protector.redact_pii(text)
        
        assert '@' not in redacted
        assert 'test' not in redacted
        assert '*' in redacted
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing."""
        text = "sensitive_data"
        hash1 = self.protector.hash_sensitive_data(text)
        hash2 = self.protector.hash_sensitive_data(text)
        
        assert hash1 == hash2
        assert len(hash1) == 16


class TestRuleEvidenceExtractor:
    """Test cases for RuleEvidenceExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = RuleEvidenceExtractor()
    
    def test_extract_evidence_hardcoded_password(self):
        """Test evidence extraction for hardcoded passwords."""
        code = "password = 'secret123'"
        evidence = self.extractor.extract_evidence(code, 'hardcoded_password')
        
        assert len(evidence) > 0
        assert 'Hardcoded password found' in evidence[0]
    
    def test_extract_evidence_eval_usage(self):
        """Test evidence extraction for eval usage."""
        code = "result = eval(user_input)"
        evidence = self.extractor.extract_evidence(code, 'eval_usage')
        
        assert len(evidence) > 0
        assert 'eval() function call' in evidence[0]
    
    def test_get_rule_explanation(self):
        """Test rule explanation generation."""
        explanation = self.extractor.get_rule_explanation('hardcoded_password')
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestExplainabilityManager:
    """Test cases for ExplainabilityManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ExplainabilityManager()
    
    def test_protect_output(self):
        """Test output protection."""
        text = "Email: test@example.com"
        protected = self.manager.protect_output(text)
        
        assert '@' not in protected
        assert '*' in protected
    
    def test_explain_vulnerability(self):
        """Test vulnerability explanation."""
        vulnerability = {
            'rule_id': 'TEST_RULE',
            'vulnerability_type': 'hardcoded_password',
            'confidence': 0.9
        }
        code = "password = 'secret'"
        
        explanation = self.manager.explain_vulnerability(vulnerability, code)
        
        assert explanation.vulnerability_id == 'TEST_RULE'
        assert explanation.confidence_score == 0.9
        assert len(explanation.rule_evidence) > 0


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis pipeline."""
        # Generate sample code
        code = """
def login(username, password):
    if password == 'admin123':
        return True
    return False
"""
        
        # Initialize components
        scanner = CodeScanner()
        extractor = CodeFeatureExtractor()
        
        # Scan code
        vulnerabilities = scanner.scan_code(code)
        assert len(vulnerabilities) > 0
        
        # Extract features
        features = extractor.extract_all_features(code)
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Test with mock model
        detector = VulnerabilityDetector(model_type='logistic_regression')
        X = np.random.rand(1, len(features))
        y = np.array([1])
        
        detector.fit(X, y)
        predictions = detector.predict(X)
        assert len(predictions) == 1
    
    def test_dataset_generation_and_training(self):
        """Test dataset generation and model training."""
        # Generate dataset
        generator = CodeDatasetGenerator({'random_seed': 42})
        dataset = generator.generate_dataset(20, 20)
        
        # Prepare features
        extractor = CodeFeatureExtractor()
        train_samples = dataset['train']['code'].tolist()
        train_labels = [1 if v == 'critical' else 0 for v in dataset['train']['severity']]
        
        X_train = extractor.fit_transform(train_samples)
        y_train = np.array(train_labels)
        
        # Train model
        detector = VulnerabilityDetector(model_type='logistic_regression')
        detector.fit(X_train, y_train)
        
        # Test prediction
        predictions = detector.predict(X_train)
        assert len(predictions) == len(y_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
