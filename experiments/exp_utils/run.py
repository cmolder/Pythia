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
    #'scooby'  : 'scooby_pref_degree',       # Default - it can be dynamic
    'sisb'    : 'sisb_pref_degree',
    'sms'     : 'sms_pref_degree',
    'spp_dev2': 'spp_pref_degree',
    'streamer': 'streamer_pref_degree',
    'triage'  : 'triage_max_allowed_degree' # Maximum - it is dynamic
    
    # Sandbox, Bingo have no degree knobs
    # Pythia has dynamic degrees by default.
}

def get_binary(**kwargs):
    """Return name of a binary
    """
    binary = (defaults.binary_base + defaults.llc_sets_suffix).format(**kwargs)
    
    return os.path.join(
        #os.path.dirname(__file__),
        binary
    )

def get_results_file(binary, traces, l1c_prefs=[], l2c_prefs=[], llc_prefs=[], 
                     l1c_pref_degrees=[], l2c_pref_degrees=[], llc_pref_degrees=[]):
    """Return name of a results file.
    """
    base_traces = '-'.join([''.join(os.path.basename(et).split('.')[:-2]) for et in traces])
    base_binary = os.path.basename(binary)
    
    bp, l1p, l2p, llp, llr, n_cores, n_sets = base_binary.split('-')
    
    # Prefetcher degrees
    l1pd = [str(d) for d in l1c_pref_degrees] if len(l1c_pref_degrees) == len(l1c_prefs) else ['na']*len(l1c_prefs)
    l2pd = [str(d) for d in l2c_pref_degrees] if len(l2c_pref_degrees) == len(l2c_prefs) else ['na']*len(l2c_prefs)
    llpd = [str(d) for d in llc_pref_degrees] if len(llc_pref_degrees) == len(llc_prefs) else ['na']*len(llc_prefs)
    
    if l1p == 'multi':
        l1p = ','.join(l1c_prefs) + '_' + ','.join(l1pd)
    if l2p == 'multi':
        l2p = ','.join(l2c_prefs) + '_' + ','.join(l2pd)
    if llp == 'multi':
        llp = ','.join(llc_prefs) + '_' + ','.join(llpd)
    
    return f'{base_traces}-{bp}-{l1p}-{l2p}-{llp}-{llr}-{n_cores}-{n_sets}.txt'

def get_prefetcher_knobs(prefetchers, pref_degrees=[], level='llc'):
    assert (pref_degrees == [] or len(pref_degrees) == len(prefetchers)), 'Must pass one degree for each prefetcher, if providing degrees'
    
    knobs = []
    for i, t in enumerate(prefetchers):
        knobs.append(f'--{level}_prefetcher_types={t}')
        
        # NOTE: Will ignore the degree knob, if the prefetcher lacks one.
        if t in pref_degree_knobs and len(pref_degrees) == len(prefetchers):
            knobs.append(f'--{pref_degree_knobs[t]}={pref_degrees[i]}')
        
    return ' '.join(knobs)