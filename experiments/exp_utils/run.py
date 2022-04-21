"""
Utility functions for running versions of Pythia fork of ChampSim.

Author: Carson Molder
"""

# Try not to import anything outside Python default libraries.
import os
from exp_utils import defaults


pref_degree_knobs = {
    'ampm'    : 'ampm_pref_degree',
    'bop'     : 'bop_pref_degree',
    'dspatch' : 'dspatch_pref_degree',
    'mlop'    : 'mlop_pref_degree',
    #'scooby'  : 'scooby_pref_degree',      # Default - it can be dynamic
    'sisb'    : 'sisb_pref_degree',
    'sms'     : 'sms_pref_degree',
    #'spp_dev2': 'spp_pref_degree',         # Has no effect on SPP.
    'streamer': 'streamer_pref_degree',
    'triage'  : 'triage_max_allowed_degree' # Maximum - it is dynamic
    
    # Sandbox, Bingo have no degree knobs
    # Pythia has dynamic degrees by default.
    # SPP's degree knob has no effect.
}

def get_llc_pref_fn(llc_prefs):
    if llc_prefs == ['no']:
        return 'no'
    elif llc_prefs == ['pc_trace']:
        return 'multi_pc_trace'
    return 'multi'

def get_l2c_pref_fn(l2c_prefs):
    if l2c_prefs == ['no']:
        return 'no'
    return 'multi'

def get_l1d_pref_fn(l1d_prefs):
    if l1d_prefs == ['no']:
        return 'no'
    return 'multi'



def get_binary(**kwargs):
    """Return name of a binary
    """
    binary = (defaults.binary_base + defaults.llc_sets_suffix).format(**kwargs)
    
    return os.path.join(
        #os.path.dirname(__file__),
        binary
    )

def get_results_file(binary, traces, l1d_prefs=[], l2c_prefs=[], llc_prefs=[], 
                     l2c_pref_degrees=[], llc_pref_degrees=[]):
    """Return name of a results file.
    """
    base_traces = '-'.join([''.join(os.path.basename(et).split('.')[:-2]) for et in traces])
    base_binary = os.path.basename(binary)
    
    bp, l1p, l2p, llp, llr, n_cores, n_sets = base_binary.split('-')
    
    # Prefetcher degrees
    l2pd = [str(d) for d in l2c_pref_degrees] if len(l2c_pref_degrees) == len(l2c_prefs) else ['na']*len(l2c_prefs)
    llpd = [str(d) for d in llc_pref_degrees] if len(llc_pref_degrees) == len(llc_prefs) else ['na']*len(llc_prefs)
    
    if l1p == 'multi':
        l1p = ','.join(l1d_prefs)
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


def _is_cloudsuite(trace):
    trace = os.path.basename(trace)
    tokens = trace.split('_')
    
    return(len(tokens) == 3 and tokens[1].startswith('phase') and tokens[2].startswith('core'))


def get_cloudsuite_knobs(traces):
    """Parse the format of the filenames to determine if the run is using
    a CloudSuite trace.
    
    (TODO: Do automatically by reading the file format).
    """
    is_cloudsuite = [_is_cloudsuite(t) for t in traces]
    
    assert all(i for i in is_cloudsuite) or all(~i for i in is_cloudsuite), 'Cannot mix CloudSuite and non-CloudSuite traces'
    
    if all(i for i in is_cloudsuite):
        return '--knob_cloudsuite=true'
    else:
        return ''

def get_output_trace_knobs(results_dir, results_file, track_pc=False, track_addr=False):
    """Get the knobs required to track per-PC and per-address
    prefetch statistics, including the toggle knob and output file path.
    """
    knobs = ''
    
    
    if track_pc:
        pc_pref_dir = os.path.join(results_dir, 'pc_pref_stats')
        os.makedirs(pc_pref_dir, exist_ok=True)
        knobs += '--measure_pc_prefetches=true '
        
    if track_addr:
        addr_pref_dir = os.path.join(results_dir, 'addr_pref_stats')
        os.makedirs(addr_pref_dir, exist_ok=True)
        knobs += '--measure_addr_prefetches=true '
    
    for level in ['l1d', 'l2c', 'llc']:
        level_results_file = results_file.replace('.txt', f'_{level}.txt')
        
        if track_pc:
            knobs += f' --pc_prefetch_file_{level}={pc_pref_dir}/{level_results_file}'
        if track_addr:
            knobs += f' --addr_prefetch_file_{level}={addr_pref_dir}/{level_results_file}'
    
    return knobs