"""
Data generation and synthesis for static code analysis.

This module provides functionality to generate synthetic datasets
for training and testing vulnerability detection models.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CodeDatasetGenerator:
    """Generates synthetic code datasets for vulnerability detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the dataset generator."""
        self.config = config or {}
        self.random_seed = self.config.get('random_seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Vulnerability patterns
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.safe_patterns = self._load_safe_patterns()
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """Load code patterns that contain vulnerabilities."""
        return {
            'hardcoded_password': [
                "password = 'admin123'",
                "user_pass = \"secretpassword\"",
                "api_key = 'sk-1234567890abcdef'",
                "secret_token = \"my_secret_token\"",
                "db_password = 'password123'"
            ],
            'eval_usage': [
                "result = eval(user_input)",
                "code = eval(expression)",
                "output = eval(f'print({x})')",
                "result = eval('2 + 2')",
                "exec(eval(user_code))"
            ],
            'sql_injection': [
                "query = f'SELECT * FROM users WHERE id = {user_id}'",
                "cursor.execute('SELECT * FROM table WHERE name = ' + name)",
                "sql = \"SELECT * FROM users WHERE username = '\" + username + \"'\"",
                "query = 'DELETE FROM users WHERE id = ' + str(user_id)",
                "cursor.execute(f\"INSERT INTO table VALUES ({value})\")"
            ],
            'shell_injection': [
                "os.system(f'rm {filename}')",
                "subprocess.call(['ls', user_input])",
                "os.system('ping ' + hostname)",
                "subprocess.run(f'echo {user_data}')",
                "os.system('cat ' + file_path)"
            ],
            'weak_crypto': [
                "hash = hashlib.md5(password.encode()).hexdigest()",
                "checksum = hashlib.sha1(data).hexdigest()",
                "import md5; md5.new(text).hexdigest()",
                "hashlib.sha1(input_string.encode()).hexdigest()",
                "md5_hash = hashlib.md5(file_content).hexdigest()"
            ],
            'insecure_random': [
                "random_number = random.random()",
                "id = random.randint(1, 1000)",
                "token = random.choice(['a', 'b', 'c'])",
                "password = ''.join(random.choices(string.ascii_letters, k=8))",
                "session_id = random.randrange(1000000)"
            ],
            'exception_handling': [
                "try:\n    risky_operation()\nexcept:\n    pass",
                "try:\n    file.read()\nexcept:\n    continue",
                "try:\n    api_call()\nexcept:\n    return None",
                "try:\n    database.query()\nexcept:\n    pass",
                "try:\n    network_request()\nexcept:\n    pass"
            ]
        }
    
    def _load_safe_patterns(self) -> List[str]:
        """Load safe code patterns."""
        return [
            "def safe_function():\n    return 'Hello World'",
            "import os\nprint(os.getcwd())",
            "def calculate(a, b):\n    return a + b",
            "class MyClass:\n    def __init__(self):\n        self.value = 0",
            "for i in range(10):\n    print(i)",
            "if condition:\n    do_something()\nelse:\n    do_other()",
            "try:\n    risky_operation()\nexcept SpecificException as e:\n    handle_error(e)",
            "import hashlib\nhash = hashlib.sha256(data.encode()).hexdigest()",
            "import secrets\nsecure_token = secrets.token_hex(32)",
            "def validate_input(user_input):\n    if not user_input:\n        raise ValueError('Invalid input')"
        ]
    
    def generate_vulnerable_code(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate code samples with vulnerabilities."""
        logger.info(f"Generating {num_samples} vulnerable code samples")
        
        samples = []
        
        for i in range(num_samples):
            # Choose vulnerability type
            vuln_type = random.choice(list(self.vulnerability_patterns.keys()))
            vuln_pattern = random.choice(self.vulnerability_patterns[vuln_type])
            
            # Generate surrounding context
            context = self._generate_context()
            
            # Combine context with vulnerability
            code = self._combine_code(context, vuln_pattern)
            
            # Determine severity
            severity = self._get_severity(vuln_type)
            
            sample = {
                'code': code,
                'vulnerability_type': vuln_type,
                'severity': severity,
                'line_number': self._find_vulnerability_line(code, vuln_pattern),
                'description': self._get_description(vuln_type),
                'confidence': random.uniform(0.7, 1.0),
                'rule_id': f'{vuln_type.upper()}_RULE'
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_safe_code(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate safe code samples."""
        logger.info(f"Generating {num_samples} safe code samples")
        
        samples = []
        
        for i in range(num_samples):
            # Choose safe pattern
            safe_pattern = random.choice(self.safe_patterns)
            
            # Generate additional context
            context = self._generate_context()
            
            # Combine context with safe pattern
            code = self._combine_code(context, safe_pattern)
            
            sample = {
                'code': code,
                'vulnerability_type': 'safe',
                'severity': 'info',
                'line_number': 1,
                'description': 'No vulnerabilities detected',
                'confidence': random.uniform(0.8, 1.0),
                'rule_id': 'SAFE_CODE'
            }
            
            samples.append(sample)
        
        return samples
    
    def _generate_context(self) -> str:
        """Generate contextual code around the main pattern."""
        context_options = [
            "def process_data(data):\n    ",
            "class DataProcessor:\n    def __init__(self):\n        ",
            "import os\nimport sys\n\n",
            "def main():\n    ",
            "if __name__ == '__main__':\n    ",
            "import logging\nlogger = logging.getLogger(__name__)\n\n",
            "def validate_user(user):\n    ",
            "class UserManager:\n    def authenticate(self, username, password):\n        "
        ]
        
        return random.choice(context_options)
    
    def _combine_code(self, context: str, pattern: str) -> str:
        """Combine context with vulnerability pattern."""
        # Simple combination - in practice, this could be more sophisticated
        if context.strip().endswith(':'):
            return context + "\n    " + pattern
        else:
            return context + "\n" + pattern
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_mapping = {
            'hardcoded_password': 'critical',
            'eval_usage': 'high',
            'sql_injection': 'critical',
            'shell_injection': 'high',
            'weak_crypto': 'medium',
            'insecure_random': 'medium',
            'exception_handling': 'low'
        }
        
        return severity_mapping.get(vuln_type, 'medium')
    
    def _get_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            'hardcoded_password': 'Hardcoded password detected in source code',
            'eval_usage': 'Use of eval() function can lead to code injection',
            'sql_injection': 'Potential SQL injection vulnerability',
            'shell_injection': 'Potential shell injection vulnerability',
            'weak_crypto': 'Use of weak cryptographic hash function',
            'insecure_random': 'Use of insecure random number generation',
            'exception_handling': 'Bare except clause detected'
        }
        
        return descriptions.get(vuln_type, 'Security vulnerability detected')
    
    def _find_vulnerability_line(self, code: str, pattern: str) -> int:
        """Find the line number where the vulnerability pattern occurs."""
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if pattern.strip() in line:
                return i
        return 1
    
    def generate_dataset(
        self,
        num_vulnerable: int = 200,
        num_safe: int = 300,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a complete dataset with train/validation/test splits."""
        logger.info(f"Generating dataset with {num_vulnerable} vulnerable and {num_safe} safe samples")
        
        # Generate samples
        vulnerable_samples = self.generate_vulnerable_code(num_vulnerable)
        safe_samples = self.generate_safe_code(num_safe)
        
        # Combine and shuffle
        all_samples = vulnerable_samples + safe_samples
        random.shuffle(all_samples)
        
        # Split dataset
        train_size = int(len(all_samples) * split_ratios[0])
        val_size = int(len(all_samples) * split_ratios[1])
        
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:train_size + val_size]
        test_samples = all_samples[train_size + val_size:]
        
        dataset = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }
        
        logger.info(f"Dataset generated: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
        """Save dataset to files."""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, samples in dataset.items():
            # Convert to DataFrame
            df = pd.DataFrame(samples)
            
            # Save as CSV
            csv_path = output_path / f'{split_name}.csv'
            df.to_csv(csv_path, index=False)
            
            # Save as JSON for easier loading
            json_path = output_path / f'{split_name}.json'
            df.to_json(json_path, orient='records', indent=2)
            
            logger.info(f"Saved {split_name} split: {len(samples)} samples to {csv_path}")
    
    def load_dataset(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """Load dataset from files."""
        from pathlib import Path
        
        data_path = Path(data_dir)
        dataset = {}
        
        for split_name in ['train', 'validation', 'test']:
            csv_path = data_path / f'{split_name}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                dataset[split_name] = df
                logger.info(f"Loaded {split_name} split: {len(df)} samples")
            else:
                logger.warning(f"Split file not found: {csv_path}")
        
        return dataset
    
    def get_dataset_stats(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {}
        
        for split_name, samples in dataset.items():
            if not samples:
                continue
            
            # Count by vulnerability type
            vuln_counts = {}
            severity_counts = {}
            
            for sample in samples:
                vuln_type = sample['vulnerability_type']
                severity = sample['severity']
                
                vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            stats[split_name] = {
                'total_samples': len(samples),
                'vulnerability_types': vuln_counts,
                'severity_distribution': severity_counts,
                'avg_confidence': np.mean([s['confidence'] for s in samples])
            }
        
        return stats
