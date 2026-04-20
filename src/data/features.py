"""
Feature engineering and data processing for static code analysis.

This module provides functionality to extract features from code
and prepare data for machine learning models.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class CodeFeatureExtractor:
    """Extracts features from Python code for ML models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the feature extractor."""
        self.config = config or {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words=None
        )
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
    
    def extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract features from AST representation."""
        features = {}
        
        try:
            tree = ast.parse(code)
            
            # Basic AST metrics
            features['ast_nodes'] = len(list(ast.walk(tree)))
            features['ast_depth'] = self._get_ast_depth(tree)
            
            # Function and class counts
            features['function_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            features['class_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            
            # Control flow complexity
            features['if_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
            features['for_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.For)])
            features['while_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.While)])
            features['try_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.Try)])
            
            # Import analysis
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            features['import_count'] = len(imports)
            
            # Dangerous function usage
            dangerous_functions = ['eval', 'exec', 'compile', 'input', 'open']
            call_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            dangerous_calls = 0
            
            for call in call_nodes:
                if isinstance(call.func, ast.Name) and call.func.id in dangerous_functions:
                    dangerous_calls += 1
            
            features['dangerous_calls'] = dangerous_calls
            
            # String literal analysis
            string_literals = [n for n in ast.walk(tree) if isinstance(n, ast.Str)]
            features['string_count'] = len(string_literals)
            features['avg_string_length'] = np.mean([len(s.s) for s in string_literals]) if string_literals else 0
            
            # Check for hardcoded secrets patterns
            features['potential_secrets'] = self._count_potential_secrets(string_literals)
            
        except SyntaxError:
            # Handle syntax errors gracefully
            features.update({
                'ast_nodes': 0,
                'ast_depth': 0,
                'function_count': 0,
                'class_count': 0,
                'if_count': 0,
                'for_count': 0,
                'while_count': 0,
                'try_count': 0,
                'import_count': 0,
                'dangerous_calls': 0,
                'string_count': 0,
                'avg_string_length': 0,
                'potential_secrets': 0
            })
        
        return features
    
    def _get_ast_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate the maximum depth of the AST."""
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_ast_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _count_potential_secrets(self, string_literals: List[ast.Str]) -> int:
        """Count potential hardcoded secrets in string literals."""
        secret_patterns = [
            r'password\s*=\s*[\'"][^\'"]+[\'"]',
            r'secret\s*=\s*[\'"][^\'"]+[\'"]',
            r'key\s*=\s*[\'"][^\'"]+[\'"]',
            r'token\s*=\s*[\'"][^\'"]+[\'"]',
            r'api_key\s*=\s*[\'"][^\'"]+[\'"]'
        ]
        
        count = 0
        for string_lit in string_literals:
            for pattern in secret_patterns:
                if re.search(pattern, string_lit.s, re.IGNORECASE):
                    count += 1
                    break
        
        return count
    
    def extract_text_features(self, code: str) -> Dict[str, Any]:
        """Extract text-based features from code."""
        lines = code.split('\n')
        
        features = {
            'line_count': len(lines),
            'char_count': len(code),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'max_line_length': max([len(line) for line in lines]) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('#')),
            'docstring_lines': self._count_docstring_lines(code)
        }
        
        # Code complexity metrics
        features['cyclomatic_complexity'] = self._calculate_cyclomatic_complexity(code)
        features['halstead_metrics'] = self._calculate_halstead_metrics(code)
        
        return features
    
    def _count_docstring_lines(self, code: str) -> int:
        """Count lines containing docstrings."""
        try:
            tree = ast.parse(code)
            docstring_lines = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if ast.get_docstring(node):
                        docstring_lines += len(ast.get_docstring(node).split('\n'))
            
            return docstring_lines
        except SyntaxError:
            return 0
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                elif isinstance(node, ast.Try):
                    complexity += len(node.handlers)
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except SyntaxError:
            return 1
    
    def _calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        try:
            tree = ast.parse(code)
            
            operators = set()
            operands = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.operator):
                    operators.add(type(node).__name__)
                elif isinstance(node, ast.Name):
                    operands.add(node.id)
                elif isinstance(node, ast.Str):
                    operands.add(f"string_{len(node.s)}")
            
            n1 = len(operators)  # Distinct operators
            n2 = len(operands)    # Distinct operands
            
            # Count total occurrences
            total_operators = sum(1 for node in ast.walk(tree) if isinstance(node, ast.operator))
            total_operands = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Name, ast.Str)))
            
            N1 = total_operators
            N2 = total_operands
            
            if n1 == 0 or n2 == 0:
                return {'volume': 0, 'difficulty': 0, 'effort': 0}
            
            volume = (N1 + N2) * np.log2(n1 + n2)
            difficulty = (n1 / 2) * (N2 / n2)
            effort = difficulty * volume
            
            return {
                'volume': volume,
                'difficulty': difficulty,
                'effort': effort
            }
        except SyntaxError:
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
    
    def extract_all_features(self, code: str) -> Dict[str, Any]:
        """Extract all available features from code."""
        features = {}
        
        # AST features
        ast_features = self.extract_ast_features(code)
        features.update(ast_features)
        
        # Text features
        text_features = self.extract_text_features(code)
        features.update(text_features)
        
        # Additional derived features
        features['code_density'] = features['char_count'] / max(features['line_count'], 1)
        features['complexity_ratio'] = features['cyclomatic_complexity'] / max(features['function_count'], 1)
        
        return features
    
    def fit_transform(self, code_samples: List[str]) -> np.ndarray:
        """Fit the feature extractor and transform code samples."""
        features_list = []
        
        for code in code_samples:
            features = self.extract_all_features(code)
            features_list.append(features)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(features_list)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Normalize numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
        df[numerical_cols] = df[numerical_cols].fillna(0)
        
        self._is_fitted = True
        
        return df.values
    
    def transform(self, code_samples: List[str]) -> np.ndarray:
        """Transform code samples using fitted extractor."""
        if not self._is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        return self.fit_transform(code_samples)


class VulnerabilityDataset:
    """Dataset class for vulnerability detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the dataset."""
        self.config = config or {}
        self.feature_extractor = CodeFeatureExtractor(config)
        self.data = pd.DataFrame()
        self.labels = np.array([])
    
    def load_from_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> None:
        """Load dataset from vulnerability data."""
        if not vulnerabilities:
            logger.warning("No vulnerabilities provided")
            return
        
        # Extract features from code snippets
        code_samples = [v.get('code_snippet', '') for v in vulnerabilities]
        features = self.feature_extractor.fit_transform(code_samples)
        
        # Create labels
        severity_mapping = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
        labels = [severity_mapping.get(v.get('severity', 'info'), 0) for v in vulnerabilities]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        self.data = pd.DataFrame(features, columns=feature_names)
        self.labels = np.array(labels)
        
        logger.info(f"Loaded dataset with {len(self.data)} samples and {features.shape[1]} features")
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train-test split of the dataset."""
        from sklearn.model_selection import train_test_split
        
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.values,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        if len(self.data) == 0:
            return {'samples': 0, 'features': 0, 'label_distribution': {}}
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        
        return {
            'samples': len(self.data),
            'features': self.data.shape[1],
            'label_distribution': label_distribution,
            'feature_stats': self.data.describe().to_dict()
        }
