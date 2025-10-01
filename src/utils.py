"""
Utility functions for Crisis Assistant
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary with configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging based on config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    log_level = config.get('system', {}).get('log_level', 'INFO')
    log_file = config.get('system', {}).get('log_file', 'crisis-assistant.log')
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('crisis-assistant')
    return logger


def validate_file_exists(file_path: str, file_type: str = "File") -> bool:
    """
    Check if a file exists and log error if not
    
    Args:
        file_path: Path to check
        file_type: Type of file for error message
        
    Returns:
        True if exists, False otherwise
    """
    path = Path(file_path)
    if not path.exists():
        logging.error(f"{file_type} not found: {file_path}")
        return False
    return True


def ensure_directory(dir_path: str) -> Path:
    """
    Ensure a directory exists, create if not
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path