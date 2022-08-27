"""
Utility functions for running versions of Pythia fork of ChampSim.

Author: Carson Molder
"""

# Try not to import anything outside Python default libraries.
from itertools import combinations, product
from typing import List, Optional, Tuple
import os

from exp_utils import defaults
from exp_utils.file import ChampsimStatsFile, ChampsimTraceDirectory

pref_degree_knobs = {
    'ampm': 'ampm_pref_degree',
    'bingo': 'bingo_max_degree', # Maximum - it is dynamic
    'bop': 'bop_pref_degree',
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


class Run():
    """Defines an individual run of a Condor sweep."""

    def __init__(self,
                 trace: ChampsimStatsFile,
                 l1d_prefetcher: Tuple[str] = ('no',),
                 l2_prefetcher: Tuple[str] = ('no',),
                 llc_prefetcher: Tuple[str] = ('no',),
                 l1d_prefetcher_degree: Tuple[int] = (0,),
                 l2_prefetcher_degree: Tuple[int] = (0,),
                 llc_prefetcher_degree: Tuple[int] = (0,),
                 llc_sets: Optional[int] = None,
                 branch_predictor: Optional[str] = None,
                 seed: int = 0,
                 pythia_features: Optional[Tuple[int]] = None,
                 pythia_level_threshold: Optional[float] = None,
                 pythia_pooling: Optional[str] = None):
        """TODO: Docstring
        """

        # Sanity checks
        assert len(l1d_prefetcher) == len(l1d_prefetcher_degree)
        assert len(l2_prefetcher) == len(l2_prefetcher_degree)
        assert len(llc_prefetcher) == len(llc_prefetcher_degree)

        self.trace = trace

        self.l1d_prefetcher = l1d_prefetcher
        self.l2_prefetcher = l2_prefetcher
        self.llc_prefetcher = llc_prefetcher

        self.l1d_prefetcher_degree = l1d_prefetcher_degree
        self.l2_prefetcher_degree = l2_prefetcher_degree
        self.llc_prefetcher_degree = llc_prefetcher_degree

        self.branch_predictor = branch_predictor
        self.seed = seed
        self.llc_sets = llc_sets

        self.pythia_features = pythia_features
        self.pythia_level_threshold = pythia_level_threshold
        self.pythia_pooling = pythia_pooling

    def run_name(self):
        """Generate a run name for the run."""

        # Prefetcher strings
        l1d_prefetcher_str = '' if self.l1d_prefetcher == ('no',) else (
            'l1dpf_'
            + ','.join(self.l1d_prefetcher)
            + '_'
            + ','.join([str(d) for d in self.l1d_prefetcher_degree]))
        l2_prefetcher_str = '' if self.l2_prefetcher == ('no',) else (
            'l2pf_'
            + ','.join(self.l2_prefetcher)
            + '_'
            + ','.join([str(d) for d in self.l2_prefetcher_degree]))
        llc_prefetcher_str = '' if self.llc_prefetcher == ('no',) else (
            'llcpf_'
            + ','.join(self.llc_prefetcher)
            + '_'
            + ','.join([str(d) for d in self.llc_prefetcher_degree]))

        seed_str = f'seed_{self.seed}'
        branch_predictor_str = f'bp_{self.branch_predictor}' if self.branch_predictor else ''
        llc_sets_str = f'llc_sets_{self.llc_sets}' if self.llc_sets else ''
        pythia_features_str = 'pythia_features_' + \
            ','.join([str(f) for f in self.pythia_features]
                     ) if self.pythia_features else ''
        pythia_level_threshold_str = f'pythia_level_threshold_{self.pythia_level_threshold}' if self.pythia_level_threshold else ''
        pythia_poooling_str = f'pythia_pooling_{self.pythia_pooling}' if self.pythia_pooling else ''

        # Return a single run name, skipping empty parameters.
        return '-'.join(x.strip() for x in (
            self.trace.full_trace,
            branch_predictor_str,
            l1d_prefetcher_str,
            l2_prefetcher_str,
            llc_prefetcher_str,
            llc_sets_str,
            pythia_features_str,
            pythia_level_threshold_str,
            pythia_poooling_str,
            seed_str
        ) if x.strip())

    def __str__(self):
        """TODO: Docstring
        """
        return self.run_name()


class RunGenerator():
    """Generate a list of runs by iterating on the options in the yml file."""

    def __init__(self, cfg):
        """TODO: Docstring
        """
        self.cfg = cfg
        self.traces = ChampsimTraceDirectory(cfg.paths.trace_dir)

    def __prefetcher_combinations(self):
        """TODO: Docstring
        """

        # For each level of the cache, generate all possible
        # combinations (hybrids) of prefetchers from the level's
        # list of prefetcher candidates. Also include a no-prefetcher
        # baseline.
        l1d_prefetchers = [
            p for h in self.cfg.l1d.hybrids
            for p in combinations(self.cfg.l1d.pref_candidates, h)
        ] + [('no', )]

        l2_prefetchers = [
            p for h in self.cfg.l2c.hybrids
            for p in combinations(self.cfg.l2c.pref_candidates, h)
        ] + [('no', )]

        llc_prefetchers = [
            p for h in self.cfg.llc.hybrids
            for p in combinations(self.cfg.llc.pref_candidates, h)
        ] + [('no', )]

        return product(
            l1d_prefetchers, l2_prefetchers, llc_prefetchers)

    def __get_valid_degrees(self, degree_candidates: List[int],
                            prefetcher: Tuple[str]):
        """Get the valid options for degree, given the prefetcher and
        the overall options for degree.

        Parameters:
            degree_candidates: A list of integers representing possible 
                degrees.
            prefetcher: A tuple of prefetcher names that make up a
                single prefetcher (hybrid) for one level of the cache.
                Can be a single tuple, to represent non-hybrid.
        """

        candidates = [(0,) for _ in range(len(prefetcher))]
        for i, pref in enumerate(prefetcher):
            # Filter out and don't sweep degree for the following:
            # 1) No prefetcher
            # 2) Prefetchers with dynamic degree schemes enabled
            # 3) Prefetchers that lack degree knobs
            if ('no' in pref
                or 'scooby' in pref and self.cfg.pythia.dyn_degree is True
                or 'bingo' in pref and self.cfg.bingo.dyn_degree is True
                or 'spp_dev2' in pref and self.cfg.spp_dev2.dyn_degree is True
                or pref not in pref_degree_knobs.keys()):
                continue
            else:
                candidates[i] = degree_candidates
        return product(*candidates)

    def __degree_combinations(self,
                              l1d_prefetcher: Tuple[str],
                              l2_prefetcher: Tuple[str],
                              llc_prefetcher: Tuple[str]):
        """TODO: Docstring
        """
        l1d_degrees = self.__get_valid_degrees(
            self.cfg.l1d.degrees, l1d_prefetcher)
        l2_degrees = self.__get_valid_degrees(
            self.cfg.l2c.degrees, l2_prefetcher)
        llc_degrees = self.__get_valid_degrees(
            self.cfg.llc.degrees, llc_prefetcher)

        return product(l1d_degrees, l2_degrees, llc_degrees)

    def __extras_combinations(self,
                              l1d_prefetcher: Tuple[str],
                              l2_prefetcher: Tuple[str],
                              llc_prefetcher: Tuple[str]):
        """Generate combinations for extra knobs, like seed and 
        prefetcher-specific knobs.

        TODO: Docstring
        """
        seeds = []
        pythia_features = []
        pythia_level_thresholds = []

        if 'seeds' in self.cfg.champsim:
            seeds = self.cfg.champsim.seeds
        if 'pythia' in self.cfg and any('scooby' in p for p in (l1d_prefetcher, l2_prefetcher, llc_prefetcher)):
            # TODO: Skip dyn_level_threshold if we have scooby_double.
            if 'dyn_level_threshold' in self.cfg.pythia:
                pythia_level_thresholds = self.cfg.pythia.dyn_level_threshold
            if 'features' in self.cfg.pythia:
                pythia_features = self.cfg.pythia.features

        # Ensure that we can still do extras combinations even if a particular
        # extra is not defined.
        # TODO: Achieve this more elegantly.
        if len(seeds) == 0:
            seeds = [None]
        if len(pythia_features) == 0:
            pythia_features = [None]
        if len(pythia_level_thresholds) == 0:
            pythia_level_thresholds = [None]

        return product(seeds, pythia_features, pythia_level_thresholds)

    def __iter__(self):
        """TODO: Docstring
        """
        for trace in self.traces:
            for l1d_prefetcher, l2_prefetcher, llc_prefetcher in self.__prefetcher_combinations():
                for l1d_degree, l2_degree, llc_degree in self.__degree_combinations(l1d_prefetcher, l2_prefetcher, llc_prefetcher):
                    for seed, pythia_features, pythia_level_threshold in self.__extras_combinations(l1d_prefetcher, l2_prefetcher, llc_prefetcher):
                        yield Run(
                            trace,
                            l1d_prefetcher=l1d_prefetcher,
                            l2_prefetcher=l2_prefetcher,
                            llc_prefetcher=llc_prefetcher,
                            l1d_prefetcher_degree=l1d_degree,
                            l2_prefetcher_degree=l2_degree,
                            llc_prefetcher_degree=llc_degree,
                            # llc_sets=self.cfg.llc.sets,
                            # branch_predictor=self.cfg.champsim.branch_pred,
                            seed=seed,
                            pythia_features=pythia_features,
                            pythia_level_threshold=pythia_level_threshold,
                        )


def get_llc_pref_fn(llc_prefs):
    if llc_prefs == ['no']:
        return 'no'
    elif llc_prefs == ['pc_trace']:
        return 'multi_pc_trace'
    elif llc_prefs == ['from_file']:
        return 'from_file'
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
        # os.path.dirname(__file__),
        binary)


def get_results_file(binary,
                     traces,
                     l1d_prefs=[],
                     l2c_prefs=[],
                     llc_prefs=[],
                     l2c_pref_degrees=[],
                     llc_pref_degrees=[]):
    """Return name of a results file.
    """
    base_traces = '-'.join(
        [''.join(os.path.basename(et).split('.')[:-2]) for et in traces])
    base_binary = os.path.basename(binary)

    bp, l1p, l2p, llp, llr, n_cores, n_sets = base_binary.split('-')

    # Prefetcher degrees
    l2pd = [
        str(d) for d in l2c_pref_degrees
    ] if len(l2c_pref_degrees) == len(l2c_prefs) else ['na'] * len(l2c_prefs)
    llpd = [
        str(d) for d in llc_pref_degrees
    ] if len(llc_pref_degrees) == len(llc_prefs) else ['na'] * len(llc_prefs)

    if l1p == 'multi':
        l1p = ','.join(l1d_prefs)
    if l2p == 'multi':
        l2p = ','.join(l2c_prefs) + '_' + ','.join(l2pd)
    if llp == 'multi':
        llp = ','.join(llc_prefs) + '_' + ','.join(llpd)

    return f'{base_traces}-{bp}-{l1p}-{l2p}-{llp}-{llr}-{n_cores}-{n_sets}.txt'


def get_prefetcher_knobs(prefetchers, pref_degrees=[], level='llc'):
    assert pref_degrees == [] \
        or len(pref_degrees) == len(prefetchers), (
        'Must pass one degree for each prefetcher, '
        'if providing degrees')

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

    return (len(tokens) == 3 and tokens[1].startswith('phase')
            and tokens[2].startswith('core'))


def get_cloudsuite_knobs(traces):
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
    else:
        return ''


def get_output_trace_knobs(results_dir,
                           results_file,
                           track_pc=False,
                           track_addr=False,
                           track_pref=False):
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
