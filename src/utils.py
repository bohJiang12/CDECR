"""
Utility functions for finetuning LLM
"""
from pathlib import Path
import yaml

def set_wkdir_as_root(cwd: Path) -> Path:
    """Set the working directory as project root

    Args:
        cwd: current working directory for executing python script

    Returns: the project root directory
    """
    if cwd.stem == 'scripts':
        return cwd.parent
    else:
        return cwd

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config