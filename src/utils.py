"""
Utility functions for finetuning LLM
"""
from typing import Dict
from pathlib import Path
import yaml
import json

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

def load_config(config_file: Path | str) -> Dict:
    """Load configuration file (as YAML) for finetuning

    Args:
        config_file: path of the configuration file

    Returns:
        a dictionary of config name and its value
    """
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config


def update_report(report_path: str | Path, new_data: Dict):
    """Update the evaluation report file (.json)

    Args:
        report_path: file path of report to be modified or created
        new_data: a dictionary of different metrics on a model
    """

    if isinstance(report_path, str):
        report = Path(report_path)
    report = report_path

    if report.exists():
        with open(report, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        existing_data.update(new_data)
    else:
        existing_data = new_data

    with open(report, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)


def display_hyperparams(title, hyperparams):
    # Determine the width of the table
    max_key_len = max(len(str(k)) for k in hyperparams.keys()) if hyperparams else 0

    # Compute max length of values while handling lists
    max_value_len = max(
        (max(map(len, map(str, v))) if isinstance(v, list) and v else len(str(v)))
        for v in hyperparams.values()
    ) if hyperparams else 0

    table_width = max(max_key_len + max_value_len + 10, len(title) + 4)  # Ensure a reasonable width

    # Print the title centered
    print("-" * table_width)
    print(f"| {title.center(table_width - 4)} |")
    print("-" * table_width)

    # Print each key-value pair
    for key, value in hyperparams.items():
        key_str = f"{key}:"

        if isinstance(value, list):
            if value:
                # Print the first element alongside the key
                print(f"| {key_str} {str(value[0]).rjust(table_width - len(key_str) - 6)} |")

                # Print the remaining elements in new lines, aligned
                for item in value[1:]:
                    print(f"| {' ' * len(key_str)} {str(item).rjust(table_width - len(key_str) - 6)} |")
            else:
                # Handle empty list case
                print(f"| {key_str} {'[]'.rjust(table_width - len(key_str) - 6)} |")
        else:
            # Print normal key-value pairs
            value_str = f"{value}".rjust(table_width - len(key_str) - 6)
            print(f"| {key_str} {value_str} |")

    print("-" * table_width)

