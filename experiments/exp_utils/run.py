"""
Utility functions for running versions of Pythia fork of ChampSim.

Author: Carson Molder
"""

# Try not to import anything outside Python default libraries.
import os
from typing import List, Optional

from exp_utils import defaults


pref_degree_knobs = {
    'ampm': 'ampm_pref_degree',
    'bingo': 'bingo_max_degree',  # Maximum - it is dynamic
    'bop': 'bop_pref_degree',
    'bop_orig': 'bop_pref_degree',
    'dspatch': 'dspatch_pref_degree',
    'mlop': 'mlop_pref_degree',
    'scooby': 'scooby_pref_degree',  # Note, it can be dynamic
    'sisb': 'sisb_pref_degree',
    'sms': 'sms_pref_degree',
    'spp_dev2': 'spp_dev2_max_degree',  # Maximum - it is dynamic
    'streamer': 'streamer_pref_degree',
    'triage': 'triage_max_allowed_degree'  # Maximum - it is dynamic

    # Sandbox, Bingo have no degree knobs
    # Pythia has dynamic degrees by default.
    # SPP's degree knob has no effect.
}


def get_llc_pref_fn(llc_prefs: List[str]) -> str:
    """Get the Champsim prefetcher knob for an LLC prefetcher.
    """
    if llc_prefs == ['no']:
        return 'no'
    elif llc_prefs == ['pc_trace']:
        return 'multi_pc_trace'
    elif llc_prefs == ['from_file']:
        return 'from_file'
    return 'multi'


def get_l2c_pref_fn(l2c_prefs: List[str]) -> str:
    """Get the Champsim prefetcher knob for an L2 prefetcher.
    """
    if l2c_prefs == ['no']:
        return 'no'
    return 'multi'


def get_l1d_pref_fn(l1d_prefs: List[str]) -> str:
    """Get the Champsim prefetcher knob for an L1D prefetcher.
    """
    if l1d_prefs == ['no']:
        return 'no'
    return 'multi'


def get_binary(**kwargs) -> str:
    """Get the name of a binary.
    """
    binary = (defaults.binary_base + defaults.llc_sets_suffix).format(**kwargs)

    return os.path.join(
        # os.path.dirname(__file__),
        binary)


def get_results_file(binary: str,
                     traces: List[str],
                     l1d_prefs: Optional[List[str]] = None,
                     l2c_prefs: Optional[List[str]] = None,
                     llc_prefs: Optional[List[str]] = None,
                     l2c_pref_degrees: Optional[List[int]] = None,
                     llc_pref_degrees: Optional[List[int]] = None) -> str:
    """Get the name of a results file.

    TODO: Support L1D prefetch degree.
    TODO: Use the Run class to get the results file name instead.
    """
    base_traces = '-'.join(
        [''.join(os.path.basename(et).split('.')[:-2]) for et in traces])
    base_binary = os.path.basename(binary)

    bpred, l1p, l2p, llp, llr, n_cores, n_sets = base_binary.split('-')

    # Prefetcher degrees
    if l2c_pref_degrees:
        assert(len(l2c_pref_degrees) == len(l2c_prefs))
        l2pd = list(map(str, l2c_pref_degrees))
    else:
        l2pd = ['0'] * len(l2c_prefs)

    if llc_pref_degrees:
        assert(len(llc_pref_degrees) == len(llc_prefs))
        llpd = list(map(str, llc_pref_degrees))
    else:
        llpd = ['0'] * len(llc_prefs)

    if l1p == 'multi':
        l1p = ','.join(l1d_prefs)
    if l2p == 'multi':
        l2p = ','.join(l2c_prefs) + '_' + ','.join(l2pd)
    if llp == 'multi':
        llp = ','.join(llc_prefs) + '_' + ','.join(llpd)

    return '-'.join((
        base_traces, bpred, l1p, l2p, llp, llr, n_cores, n_sets)) + '.txt'


def get_prefetcher_knobs(prefetchers: List[str],
                         pref_degrees: Optional[List[int]] = None,
                         level: str = 'llc') -> str:
    """Get the knobs required for prefetchers.
    """
    assert (not pref_degrees or len(pref_degrees) == len(prefetchers)), (
        'Must pass one degree for each prefetcher, if providing degrees')

    knobs = []
    for i, pref in enumerate(prefetchers):
        knobs.append(f'--{level}_prefetcher_types={pref}')

        # NOTE: Will ignore the degree knob, if the prefetcher lacks one.
        if (pref_degrees
            and pref in pref_degree_knobs
            and len(pref_degrees) == len(prefetchers)):
            knobs.append(f'--{pref_degree_knobs[pref]}={pref_degrees[i]}')

    return ' '.join(knobs)


def _is_cloudsuite(trace: str) -> bool:
    """Helper function that determines if a trace is from Cloudsuite.
    """
    trace = os.path.basename(trace)
    tokens = trace.split('_')

    return (len(tokens) == 3
            and tokens[1].startswith('phase')
            and tokens[2].startswith('core'))


def get_cloudsuite_knobs(traces: List[str]) -> str:
    """Parse the format of the filenames to determine if the run is
    using a CloudSuite trace.

    (TODO: Do automatically by reading the file format).
    """
    is_cloudsuite = [_is_cloudsuite(t) for t in traces]

    assert all(i for i in is_cloudsuite) \
        or all(~i for i in is_cloudsuite), (
        'Cannot mix CloudSuite and non-CloudSuite traces')

    if all(i for i in is_cloudsuite):
        return '--knob_cloudsuite=true'
    return ''


def get_output_trace_knobs(results_dir: str,
                           results_file: str,
                           track_pc: bool = False,
                           track_addr: bool = False,
                           track_pref: bool = False) -> str:
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

    if track_pref:
        pref_trace_dir = os.path.join(results_dir, 'pref_traces')
        os.makedirs(pref_trace_dir, exist_ok=True)
        knobs += '--dump_prefetch_trace=true '

    for level in ['l1d', 'l2c', 'llc']:
        level_results_file = results_file.replace('.txt', f'_{level}.txt')

        if track_pc:
            knobs += (f' --pc_prefetch_file_{level}='
                      f'{pc_pref_dir}/{level_results_file}')
        if track_addr:
            knobs += (f' --addr_prefetch_file_{level}='
                      f'{addr_pref_dir}/{level_results_file}')
        if track_pref and level == 'llc':
            knobs += (f' --prefetch_trace_{level}='
                      f'{pref_trace_dir}/'
                      f'{level_results_file.replace(".txt", ".gz")}')

    return knobs


def get_pc_trace_knobs(pc_trace_llc: bool = False,
                       pc_trace_credit: bool = False,
                       pc_trace_invoke_all: bool = False) -> str:
    """Get the knobs for recording the PC trace.

    Parameters:
        pc_trace_llc: Whether to record the LLC PC trace.
        pc_trace_credit: Whether to credit other prefetchers besides
            the one for the particular PC.
        pc_trace_invoke_all: Whether to invoke all prefetchers or just
            the one for the particular PC.

    Returns:
        knobs: A list of knobs to pass into ChampSim.
    """
    if pc_trace_llc:
        return (
            f' --pc_trace_llc={pc_trace_llc}'
            f' --pc_trace_credit_prefetch='
            f'{str(pc_trace_credit).lower()}'
            f' --pc_trace_invoke_all='
            f'{str(pc_trace_invoke_all).lower()}'
        )
    return ''

def get_prefetch_trace_knobs(prefetch_trace_llc: bool = False) -> str:
    """Get the knobs for recording the prefetch trace.

    Parameters:
        prefetch_trace_llc: Whether to record the LLC prefetch trace.

    Returns:
        knobs: A list of knobs to pass into ChampSim.
    """
    if prefetch_trace_llc:
        return f' --prefetch_trace_llc={prefetch_trace_llc}'
    return ''
