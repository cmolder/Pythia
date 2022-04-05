"""
Utility functions for running versions of Pythia fork of ChampSim.

Author: Carson Molder
"""
import os

from exp_utils import defaults

def get_binary(**kwargs):
    binary = (defaults.binary_base + defaults.llc_sets_suffix).format(**kwargs)
    
    return os.path.join(
        #os.path.dirname(__file__),
        binary
    )