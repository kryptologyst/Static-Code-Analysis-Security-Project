"""
Utility functions and helpers for the static code analysis project.

This module provides common utilities used across the project.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    import yaml
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed {self.operation_name} in {format_duration(duration)}")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def validate_file_extension(file_path: Union[str, Path], allowed_extensions: List[str]) -> bool:
    """Validate file extension."""
    file_path = Path(file_path)
    return file_path.suffix.lower() in [ext.lower() for ext in allowed_extensions]


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing or replacing invalid characters."""
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'untitled'
    
    return sanitized


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary with dot notation keys."""
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max."""
    return max(min_value, min(value, max_value))


def is_valid_email(email: str) -> bool:
    """Check if email address is valid."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    import re
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return re.match(pattern, url) is not None


def truncate_string(text: str, max_length: int, suffix: str = '...') -> str:
    """Truncate string to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Get hash of file content."""
    import hashlib
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def create_progress_bar(total: int, description: str = "Processing") -> Any:
    """Create a progress bar for long operations."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=description, unit="items")
    except ImportError:
        logger.warning("tqdm not available, using simple progress tracking")
        return None


def log_memory_usage() -> None:
    """Log current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage: {memory_mb:.1f} MB")
    except ImportError:
        logger.debug("psutil not available for memory monitoring")


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    import sys
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'system': platform.system(),
        'release': platform.release()
    }
    
    # Add PyTorch info if available
    try:
        import torch
        info['pytorch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['cuda_device_count'] = torch.cuda.device_count()
    except ImportError:
        info['pytorch_available'] = False
    
    return info


def print_system_info() -> None:
    """Print system information."""
    info = get_system_info()
    
    logger.info("System Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate configuration has required keys."""
    missing_keys = []
    
    def check_keys(d: Dict[str, Any], keys: List[str], prefix: str = ""):
        for key in keys:
            if '.' in key:
                main_key, sub_key = key.split('.', 1)
                if main_key in d and isinstance(d[main_key], dict):
                    check_keys(d[main_key], [sub_key], f"{prefix}{main_key}.")
                else:
                    missing_keys.append(f"{prefix}{key}")
            else:
                if key not in d:
                    missing_keys.append(f"{prefix}{key}")
    
    check_keys(config, required_keys)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def create_backup(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """Create backup of file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_filename
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    logger.info(f"Created backup: {backup_path}")
    return backup_path
