"""
Utility functions for finetuning LLM
"""

import yaml


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config