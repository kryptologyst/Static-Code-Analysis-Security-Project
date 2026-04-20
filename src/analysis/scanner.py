"""
Core analysis modules for static code analysis.

This module provides the main scanning and analysis functionality
for detecting security vulnerabilities and code quality issues.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Represents a detected vulnerability or code issue."""
    
    line_number: int
    column_number: int
    vulnerability_type: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    description: str
    code_snippet: str
    confidence: float
    rule_id: str
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary."""
        return {
            'line_number': self.line_number,
            'column_number': self.column_number,
            'vulnerability_type': self.vulnerability_type,
            'severity': self.severity,
            'description': self.description,
            'code_snippet': self.code_snippet,
            'confidence': self.confidence,
            'rule_id': self.rule_id,
            'file_path': self.file_path
        }


class ASTAnalyzer:
    """Analyzes Python code using Abstract Syntax Trees."""
    
    def __init__(self) -> None:
        """Initialize the AST analyzer."""
        self.vulnerability_patterns = self._load_vulnerability_patterns()
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load vulnerability detection patterns."""
        return {
            'hardcoded_password': {
                'pattern': r'password\s*=\s*[\'"][^\'"]+[\'"]',
                'severity': 'critical',
                'description': 'Hardcoded password detected'
            },
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'severity': 'high',
                'description': 'Use of eval() function detected'
            },
            'exec_usage': {
                'pattern': r'\bexec\s*\(',
                'severity': 'high',
                'description': 'Use of exec() function detected'
            },
            'sql_injection': {
                'pattern': r'execute\s*\(\s*["\'].*%.*["\']',
                'severity': 'critical',
                'description': 'Potential SQL injection vulnerability'
            },
            'shell_injection': {
                'pattern': r'os\.system\s*\(|subprocess\.call\s*\(',
                'severity': 'high',
                'description': 'Potential shell injection vulnerability'
            },
            'weak_crypto': {
                'pattern': r'md5\s*\(|sha1\s*\(',
                'severity': 'medium',
                'description': 'Use of weak cryptographic hash function'
            },
            'insecure_random': {
                'pattern': r'random\.random\s*\(|random\.randint\s*\(',
                'severity': 'medium',
                'description': 'Use of insecure random number generation'
            }
        }
    
    def analyze_ast(self, code: str, file_path: Optional[str] = None) -> List[Vulnerability]:
        """Analyze code using AST and pattern matching."""
        vulnerabilities = []
        
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            # Pattern-based analysis
            vulnerabilities.extend(self._pattern_analysis(lines, file_path))
            
            # AST-based analysis
            vulnerabilities.extend(self._ast_analysis(tree, lines, file_path))
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            vulnerabilities.append(Vulnerability(
                line_number=e.lineno or 0,
                column_number=e.offset or 0,
                vulnerability_type='syntax_error',
                severity='medium',
                description=f'Syntax error: {e.msg}',
                code_snippet=lines[e.lineno - 1] if e.lineno else '',
                confidence=1.0,
                rule_id='SYNTAX_ERROR',
                file_path=file_path
            ))
        
        return vulnerabilities
    
    def _pattern_analysis(self, lines: List[str], file_path: Optional[str]) -> List[Vulnerability]:
        """Perform pattern-based vulnerability analysis."""
        vulnerabilities = []
        
        for line_num, line in enumerate(lines, 1):
            for rule_id, pattern_info in self.vulnerability_patterns.items():
                if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                    vulnerabilities.append(Vulnerability(
                        line_number=line_num,
                        column_number=0,
                        vulnerability_type=rule_id,
                        severity=pattern_info['severity'],
                        description=pattern_info['description'],
                        code_snippet=line.strip(),
                        confidence=0.8,  # Pattern-based confidence
                        rule_id=rule_id.upper(),
                        file_path=file_path
                    ))
        
        return vulnerabilities
    
    def _ast_analysis(self, tree: ast.AST, lines: List[str], file_path: Optional[str]) -> List[Vulnerability]:
        """Perform AST-based vulnerability analysis."""
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                vulnerabilities.extend(self._analyze_function_call(node, lines, file_path))
            
            # Check for exception handling issues
            if isinstance(node, ast.Try):
                vulnerabilities.extend(self._analyze_exception_handling(node, lines, file_path))
            
            # Check for resource management issues
            if isinstance(node, ast.With):
                vulnerabilities.extend(self._analyze_resource_management(node, lines, file_path))
        
        return vulnerabilities
    
    def _analyze_function_call(self, node: ast.Call, lines: List[str], file_path: Optional[str]) -> List[Vulnerability]:
        """Analyze function calls for security issues."""
        vulnerabilities = []
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check for dangerous functions
            dangerous_functions = {
                'eval': ('high', 'Use of eval() function'),
                'exec': ('high', 'Use of exec() function'),
                'compile': ('medium', 'Use of compile() function'),
                'input': ('low', 'Use of input() function without validation')
            }
            
            if func_name in dangerous_functions:
                severity, description = dangerous_functions[func_name]
                vulnerabilities.append(Vulnerability(
                    line_number=node.lineno,
                    column_number=node.col_offset,
                    vulnerability_type=f'dangerous_function_{func_name}',
                    severity=severity,
                    description=description,
                    code_snippet=lines[node.lineno - 1] if node.lineno else '',
                    confidence=0.9,
                    rule_id=f'DANGEROUS_FUNC_{func_name.upper()}',
                    file_path=file_path
                ))
        
        return vulnerabilities
    
    def _analyze_exception_handling(self, node: ast.Try, lines: List[str], file_path: Optional[str]) -> List[Vulnerability]:
        """Analyze exception handling patterns."""
        vulnerabilities = []
        
        # Check for bare except clauses
        for handler in node.handlers:
            if handler.type is None:  # bare except
                vulnerabilities.append(Vulnerability(
                    line_number=handler.lineno,
                    column_number=handler.col_offset,
                    vulnerability_type='bare_except',
                    severity='medium',
                    description='Bare except clause detected',
                    code_snippet=lines[handler.lineno - 1] if handler.lineno else '',
                    confidence=0.7,
                    rule_id='BARE_EXCEPT',
                    file_path=file_path
                ))
        
        return vulnerabilities
    
    def _analyze_resource_management(self, node: ast.With, lines: List[str], file_path: Optional[str]) -> List[Vulnerability]:
        """Analyze resource management patterns."""
        vulnerabilities = []
        
        # This is a placeholder for more sophisticated resource analysis
        # In a real implementation, you would check for proper resource cleanup
        
        return vulnerabilities


class CodeScanner:
    """Main code scanner that orchestrates the analysis process."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the code scanner."""
        self.config = config or {}
        self.ast_analyzer = ASTAnalyzer()
        self._setup_random_seed()
    
    def _setup_random_seed(self) -> None:
        """Set up deterministic random seeding."""
        seed = self.config.get('random_seed', 42)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    def scan_code(self, code: str, file_path: Optional[str] = None) -> List[Vulnerability]:
        """Scan code for vulnerabilities."""
        logger.info(f"Scanning code{' from ' + file_path if file_path else ''}")
        
        vulnerabilities = []
        
        # AST-based analysis
        vulnerabilities.extend(self.ast_analyzer.analyze_ast(code, file_path))
        
        # Additional analysis can be added here
        # - Complexity analysis
        # - Dependency analysis
        # - Security pattern matching
        
        # Sort by severity and line number
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        vulnerabilities.sort(key=lambda v: (severity_order.get(v.severity, 5), v.line_number))
        
        logger.info(f"Found {len(vulnerabilities)} vulnerabilities")
        return vulnerabilities
    
    def scan_file(self, file_path: Union[str, Path]) -> List[Vulnerability]:
        """Scan a file for vulnerabilities."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix != '.py':
            logger.warning(f"Non-Python file: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.scan_code(code, str(file_path))
        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            return []
    
    def scan_directory(self, directory_path: Union[str, Path]) -> Dict[str, List[Vulnerability]]:
        """Scan all Python files in a directory."""
        directory_path = Path(directory_path)
        results = {}
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        python_files = list(directory_path.rglob('*.py'))
        logger.info(f"Found {len(python_files)} Python files to scan")
        
        for file_path in python_files:
            vulnerabilities = self.scan_file(file_path)
            if vulnerabilities:
                results[str(file_path)] = vulnerabilities
        
        return results
    
    def get_summary_stats(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Get summary statistics for vulnerabilities."""
        if not vulnerabilities:
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
        
        df = pd.DataFrame([v.to_dict() for v in vulnerabilities])
        
        return {
            'total': len(vulnerabilities),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_type': df['vulnerability_type'].value_counts().to_dict(),
            'files_affected': df['file_path'].nunique() if 'file_path' in df.columns else 0
        }
