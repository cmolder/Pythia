"""
Utility functions for running versions of Pythia fork of ChampSim.

Author: Carson Molder
"""
import os
from exp_utils import defaults


pref_degree_knobs = {
    'ampm'    : 'ampm_pref_degree',
    'bop'     : 'bop_pref_degree',
    'dspatch' : 'dspatch_pref_degree',
    'mlop'    : 'mlop_pref_degree',
    'scooby'  : 'scooby_pref_degree',       # Default - it can be dynamic
    'sisb'    : 'sisb_pref_degree',
    'sms'     : 'sms_pref_degree',
    'spp_dev2': 'spp_pref_degree',
    'streamer': 'streamer_pref_degree',
    'triage'  : 'triage_max_allowed_degree' # Maximum - it is dynamic
    
    # Sandbox, Bingo have no degree knobs
}

def get_binary(**kwargs):
    binary = (defaults.binary_base + defaults.llc_sets_suffix).format(**kwargs)
    
    return os.path.join(
        #os.path.dirname(__file__),
        binary
    )

def get_prefetcher_knobs(prefetchers, pref_degrees=[], level='llc'):
    assert (pref_degrees == [] or len(pref_degrees) == len(prefetchers)), 'Must pass one degree for each prefetcher, if providing degrees'
    
    knobs = []
    for i, t in enumerate(prefetchers):
        knobs.append(f'--{level}_prefetcher_types={t}')
        
        # NOTE: Will ignore the degree knob, if the prefetcher lacks one.
        if t in pref_degree_knobs:
            knobs.append(f'--{pref_degree_knobs[t]}={pref_degrees[i]}')
        
    return ' '.join(knobs)