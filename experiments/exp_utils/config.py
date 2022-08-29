"""
Utility functions for reading experiment/sweep configs (yml files).

Author: Carson Molder
"""

import yaml
import attrdict

def read_config(config_path) -> attrdict.AttrDict:
    """Read an experiment/sweep config yml file.

    Parameters:
        config_path: Path to the config yml file.

    Returns:
        config: A dictionary of the config values.
    """
    with open(config_path, 'r') as config_f:
        config = yaml.safe_load(config_f)
    return attrdict.AttrDict(config)
