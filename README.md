# Static Code Analysis Security Project

## Overview
This project implements a comprehensive static code analysis system for detecting security vulnerabilities, code quality issues, and potential bugs in Python codebases. It combines traditional rule-based analysis with machine learning approaches to provide accurate vulnerability detection and prioritization.

**DISCLAIMER**: This is a defensive research and educational demonstration tool. It is not intended for production security operations and may contain inaccuracies. Use only for legitimate security research and code quality improvement purposes.

## Features
- **AST-based Analysis**: Deep code structure analysis using Abstract Syntax Trees
- **Rule-based Detection**: Comprehensive security vulnerability patterns
- **ML-powered Prioritization**: Machine learning models for bug triage and priority ranking
- **Interactive Demo**: Streamlit-based web interface for code analysis
- **Explainable Results**: SHAP-based explanations for vulnerability detections
- **Privacy Protection**: Automatic PII detection and redaction

## Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
from src.analysis.scanner import CodeScanner
from src.models.vulnerability_detector import VulnerabilityDetector

# Initialize scanner
scanner = CodeScanner()
detector = VulnerabilityDetector()

# Analyze code
code = """
def login(username, password):
    if password == 'admin123':  # Hardcoded password
        return True
    return False
"""

results = scanner.scan_code(code)
vulnerabilities = detector.predict_vulnerabilities(results)
```

### Demo Interface
```bash
streamlit run demo/app.py
```

## Project Structure
```
src/
├── analysis/          # Core analysis modules
├── data/             # Data processing and feature engineering
├── models/           # ML models and vulnerability detection
├── evaluation/       # Metrics and evaluation tools
├── visualization/    # Plotting and visualization utilities
└── utils/           # Common utilities and helpers

configs/              # Configuration files
data/                # Sample datasets and generated data
scripts/             # Training and evaluation scripts
tests/               # Test suite
assets/              # Generated plots and results
demo/                # Streamlit demo application
```

## Training Models

### Generate Dataset and Train
```bash
python scripts/train_vulnerability_detector.py --config configs/default.yaml
```

### Evaluate Models
```bash
python scripts/evaluate_models.py --model-path models/vulnerability_detector.pkl --generate-plots
```

## Configuration

The system uses YAML configuration files for all settings:

```yaml
analysis:
  max_file_size: 1000000
  supported_extensions: [".py"]
  
models:
  vulnerability_detector:
    model_type: "gradient_boosting"
    features: ["ast_features", "text_features", "complexity_metrics"]
    
evaluation:
  test_split: 0.2
  cv_folds: 5
  metrics: ["precision@k", "recall", "f1_score"]
```

## Dataset Schemas

### Code Samples
- **Input**: Python source code files or code snippets
- **Features**: AST nodes, control flow graphs, text embeddings
- **Labels**: Vulnerability types, severity levels, priority scores

### Vulnerability Types
- **Security**: SQL injection, XSS, hardcoded secrets, unsafe deserialization
- **Quality**: Code smells, complexity issues, maintainability problems
- **Bugs**: Logic errors, exception handling, resource leaks

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Metrics and Limitations

### Key Metrics
- **Precision@K**: Accuracy of top-K vulnerability predictions
- **Vulnerability Recall**: Detection rate for different vulnerability types
- **False Positive Rate**: Rate of incorrect vulnerability flags
- **Processing Speed**: Analysis time per line of code

### Limitations
- Analysis is static only (no runtime behavior)
- Limited to Python codebases
- May miss context-dependent vulnerabilities
- Requires manual verification of critical findings

## Security Notice

This tool is designed for defensive security research and educational purposes only. It should not be used for:
- Unauthorized security testing
- Malicious code analysis
- Production security operations without proper validation

Always ensure you have proper authorization before analyzing any codebase.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run linting and tests: `pre-commit run --all-files`
5. Submit a pull request

## License

MIT License - See LICENSE file for details# Static-Code-Analysis-Security-Project
