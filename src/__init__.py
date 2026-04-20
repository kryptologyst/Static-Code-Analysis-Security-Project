"""
Package initialization for the static code analysis project.
"""

__version__ = "1.0.0"
__author__ = "Security Research Team"
__email__ = "research@example.com"
__description__ = "Static Code Analysis for Security Vulnerability Detection"

# Import main components for easy access
from .analysis.scanner import CodeScanner, Vulnerability, ASTAnalyzer
from .models.vulnerability_detector import VulnerabilityDetector, VulnerabilityRanker
from .data.features import CodeFeatureExtractor, VulnerabilityDataset
from .data.generator import CodeDatasetGenerator
from .evaluation.metrics import VulnerabilityEvaluator
from .utils.explainability import ExplainabilityManager, PIIProtector
from .utils.helpers import setup_logging, set_random_seed, get_device

__all__ = [
    # Core analysis
    'CodeScanner',
    'Vulnerability', 
    'ASTAnalyzer',
    
    # Models
    'VulnerabilityDetector',
    'VulnerabilityRanker',
    
    # Data processing
    'CodeFeatureExtractor',
    'VulnerabilityDataset',
    'CodeDatasetGenerator',
    
    # Evaluation
    'VulnerabilityEvaluator',
    
    # Utilities
    'ExplainabilityManager',
    'PIIProtector',
    'setup_logging',
    'set_random_seed',
    'get_device',
]
