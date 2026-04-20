"""
Explainability and safety features for static code analysis.

This module provides SHAP-based explanations, rule evidence,
PII protection, and audit logging capabilities.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Represents explanation results for a vulnerability detection."""
    
    vulnerability_id: str
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    rule_evidence: List[str] = None
    confidence_score: float = 0.0
    explanation_text: str = ""


class PIIProtector:
    """Protects personally identifiable information in code and outputs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the PII protector."""
        self.config = config or {}
        self.pii_patterns = self._load_pii_patterns()
        self.redaction_char = self.config.get('redaction_char', '*')
    
    def _load_pii_patterns(self) -> Dict[str, str]:
        """Load PII detection patterns."""
        return {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'mac_address': r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
            'username': r'\buser[name]?\s*=\s*[\'"][^\'"]+[\'"]',
            'password': r'\bpass[word]?\s*=\s*[\'"][^\'"]+[\'"]'
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII patterns in text."""
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            def replace_match(match):
                original = match.group(0)
                if len(original) <= 4:
                    return self.redaction_char * len(original)
                else:
                    return original[:2] + self.redaction_char * (len(original) - 4) + original[-2:]
            
            redacted_text = re.sub(pattern, replace_match, redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    def hash_sensitive_data(self, text: str, salt: str = "") -> str:
        """Hash sensitive data for consistent identification."""
        return hashlib.sha256((text + salt).encode()).hexdigest()[:16]


class RuleEvidenceExtractor:
    """Extracts evidence for rule-based vulnerability detections."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the rule evidence extractor."""
        self.config = config or {}
        self.rule_patterns = self._load_rule_patterns()
    
    def _load_rule_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load rule patterns and their evidence extraction logic."""
        return {
            'hardcoded_password': {
                'pattern': r'password\s*=\s*[\'"][^\'"]+[\'"]',
                'evidence_template': 'Hardcoded password found: {match}',
                'severity': 'critical'
            },
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'evidence_template': 'Dangerous eval() function call detected',
                'severity': 'high'
            },
            'sql_injection': {
                'pattern': r'execute\s*\(\s*["\'].*%.*["\']',
                'evidence_template': 'Potential SQL injection: {match}',
                'severity': 'critical'
            },
            'shell_injection': {
                'pattern': r'os\.system\s*\(|subprocess\.call\s*\(',
                'evidence_template': 'Shell command execution detected: {match}',
                'severity': 'high'
            },
            'weak_crypto': {
                'pattern': r'md5\s*\(|sha1\s*\(',
                'evidence_template': 'Weak cryptographic hash function: {match}',
                'severity': 'medium'
            },
            'bare_except': {
                'pattern': r'except\s*:',
                'evidence_template': 'Bare except clause detected',
                'severity': 'low'
            }
        }
    
    def extract_evidence(self, code: str, vulnerability_type: str) -> List[str]:
        """Extract evidence for a specific vulnerability type."""
        evidence = []
        
        if vulnerability_type not in self.rule_patterns:
            return evidence
        
        rule_info = self.rule_patterns[vulnerability_type]
        pattern = rule_info['pattern']
        template = rule_info['evidence_template']
        
        # Find all matches
        matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            # Get line number
            line_num = code[:match.start()].count('\n') + 1
            
            # Format evidence
            evidence_text = template.format(match=match.group(0).strip())
            evidence.append(f"Line {line_num}: {evidence_text}")
        
        return evidence
    
    def get_rule_explanation(self, vulnerability_type: str) -> str:
        """Get explanation for a rule-based detection."""
        explanations = {
            'hardcoded_password': 'Hardcoded passwords in source code are a critical security risk as they can be easily discovered by attackers.',
            'eval_usage': 'The eval() function executes arbitrary Python code, making it vulnerable to code injection attacks.',
            'sql_injection': 'String concatenation in SQL queries can lead to SQL injection vulnerabilities.',
            'shell_injection': 'Direct execution of shell commands with user input can lead to command injection.',
            'weak_crypto': 'MD5 and SHA1 are cryptographically broken and should not be used for security purposes.',
            'bare_except': 'Bare except clauses can hide important errors and make debugging difficult.'
        }
        
        return explanations.get(vulnerability_type, 'Security vulnerability detected.')


class SHAPExplainer:
    """Provides SHAP-based explanations for ML model predictions."""
    
    def __init__(self, model, feature_names: List[str], config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the SHAP explainer."""
        self.model = model
        self.feature_names = feature_names
        self.config = config or {}
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self) -> None:
        """Set up the appropriate SHAP explainer based on model type."""
        try:
            # Try TreeExplainer first (for tree-based models)
            if hasattr(self.model, 'tree_'):
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'estimators_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self._get_background_data())
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _get_background_data(self, n_samples: int = 100) -> np.ndarray:
        """Get background data for KernelExplainer."""
        # Generate random background data
        n_features = len(self.feature_names)
        return np.random.normal(0, 1, (n_samples, n_features))
    
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using SHAP."""
        if self.explainer is None:
            return {'error': 'SHAP explainer not available'}
        
        try:
            # Get SHAP values
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
            else:
                shap_values = self.explainer(X[instance_idx:instance_idx+1])
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Get feature importance
            feature_importance = dict(zip(self.feature_names, shap_values[0]))
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'shap_values': shap_values[0].tolist(),
                'feature_importance': dict(sorted_features[:10]),  # Top 10 features
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                'prediction': self.model.predict(X[instance_idx:instance_idx+1])[0]
            }
        
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def explain_batch(self, X: np.ndarray, max_samples: int = 10) -> List[Dict[str, Any]]:
        """Explain multiple predictions."""
        explanations = []
        
        for i in range(min(len(X), max_samples)):
            explanation = self.explain_prediction(X, i)
            explanations.append(explanation)
        
        return explanations


class AuditLogger:
    """Provides audit logging for security analysis activities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the audit logger."""
        self.config = config or {}
        self.log_file = self.config.get('audit_log_file', 'logs/audit.log')
        self.max_log_size = self.config.get('max_log_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = self.config.get('backup_count', 5)
        
        # Ensure log directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Set up audit logging."""
        logger = logging.getLogger('audit')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False
        
        self.audit_logger = logger
    
    def log_analysis_start(self, file_path: str, user_id: Optional[str] = None) -> None:
        """Log the start of an analysis."""
        self.audit_logger.info(
            f"ANALYSIS_START - File: {file_path}, User: {user_id or 'anonymous'}, "
            f"Timestamp: {datetime.now().isoformat()}"
        )
    
    def log_analysis_complete(
        self,
        file_path: str,
        vulnerabilities_found: int,
        user_id: Optional[str] = None
    ) -> None:
        """Log the completion of an analysis."""
        self.audit_logger.info(
            f"ANALYSIS_COMPLETE - File: {file_path}, Vulnerabilities: {vulnerabilities_found}, "
            f"User: {user_id or 'anonymous'}, Timestamp: {datetime.now().isoformat()}"
        )
    
    def log_vulnerability_detection(
        self,
        vulnerability_type: str,
        severity: str,
        file_path: str,
        line_number: int,
        user_id: Optional[str] = None
    ) -> None:
        """Log a vulnerability detection."""
        self.audit_logger.info(
            f"VULNERABILITY_DETECTED - Type: {vulnerability_type}, Severity: {severity}, "
            f"File: {file_path}, Line: {line_number}, User: {user_id or 'anonymous'}, "
            f"Timestamp: {datetime.now().isoformat()}"
        )
    
    def log_model_prediction(
        self,
        model_type: str,
        prediction: Any,
        confidence: float,
        user_id: Optional[str] = None
    ) -> None:
        """Log a model prediction."""
        self.audit_logger.info(
            f"MODEL_PREDICTION - Model: {model_type}, Prediction: {prediction}, "
            f"Confidence: {confidence:.3f}, User: {user_id or 'anonymous'}, "
            f"Timestamp: {datetime.now().isoformat()}"
        )
    
    def log_pii_detection(self, pii_type: str, file_path: str, user_id: Optional[str] = None) -> None:
        """Log PII detection."""
        self.audit_logger.warning(
            f"PII_DETECTED - Type: {pii_type}, File: {file_path}, "
            f"User: {user_id or 'anonymous'}, Timestamp: {datetime.now().isoformat()}"
        )
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = 'medium',
        user_id: Optional[str] = None
    ) -> None:
        """Log a general security event."""
        log_level = logging.WARNING if severity in ['high', 'critical'] else logging.INFO
        
        self.audit_logger.log(
            log_level,
            f"SECURITY_EVENT - Type: {event_type}, Description: {description}, "
            f"Severity: {severity}, User: {user_id or 'anonymous'}, "
            f"Timestamp: {datetime.now().isoformat()}"
        )


class ExplainabilityManager:
    """Manages all explainability and safety features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the explainability manager."""
        self.config = config or {}
        self.pii_protector = PIIProtector(config)
        self.rule_extractor = RuleEvidenceExtractor(config)
        self.audit_logger = AuditLogger(config)
        self.shap_explainer = None
    
    def set_model(self, model, feature_names: List[str]) -> None:
        """Set the ML model for SHAP explanations."""
        self.shap_explainer = SHAPExplainer(model, feature_names, self.config)
    
    def explain_vulnerability(
        self,
        vulnerability: Dict[str, Any],
        code: str,
        model_features: Optional[np.ndarray] = None
    ) -> ExplanationResult:
        """Provide comprehensive explanation for a vulnerability."""
        
        vuln_id = vulnerability.get('rule_id', 'unknown')
        vuln_type = vulnerability.get('vulnerability_type', 'unknown')
        
        # Extract rule evidence
        rule_evidence = self.rule_extractor.extract_evidence(code, vuln_type)
        
        # Get rule explanation
        explanation_text = self.rule_extractor.get_rule_explanation(vuln_type)
        
        # Get SHAP explanation if available
        feature_importance = {}
        shap_values = None
        
        if self.shap_explainer and model_features is not None:
            try:
                shap_explanation = self.shap_explainer.explain_prediction(model_features)
                if 'error' not in shap_explanation:
                    feature_importance = shap_explanation.get('feature_importance', {})
                    shap_values = np.array(shap_explanation.get('shap_values', []))
            except Exception as e:
                logger.warning(f"Could not generate SHAP explanation: {e}")
        
        # Create explanation result
        explanation = ExplanationResult(
            vulnerability_id=vuln_id,
            feature_importance=feature_importance,
            shap_values=shap_values,
            rule_evidence=rule_evidence,
            confidence_score=vulnerability.get('confidence', 0.0),
            explanation_text=explanation_text
        )
        
        # Log the explanation
        self.audit_logger.log_model_prediction(
            'vulnerability_explanation',
            vuln_type,
            vulnerability.get('confidence', 0.0)
        )
        
        return explanation
    
    def protect_output(self, text: str) -> str:
        """Protect PII in output text."""
        # Detect PII
        detected_pii = self.pii_protector.detect_pii(text)
        
        # Log PII detection
        for pii_type in detected_pii:
            self.audit_logger.log_pii_detection(pii_type, 'output_text')
        
        # Redact PII
        protected_text = self.pii_protector.redact_pii(text)
        
        return protected_text
    
    def get_explanation_summary(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Get summary of multiple explanations."""
        if not explanations:
            return {}
        
        # Aggregate feature importance
        all_features = {}
        for exp in explanations:
            for feature, importance in exp.feature_importance.items():
                all_features[feature] = all_features.get(feature, 0) + abs(importance)
        
        # Sort by importance
        sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_explanations': len(explanations),
            'top_features': dict(sorted_features[:10]),
            'avg_confidence': np.mean([exp.confidence_score for exp in explanations]),
            'rule_evidence_count': sum(len(exp.rule_evidence or []) for exp in explanations)
        }
