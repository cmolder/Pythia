"""
Utility functions for setting up Condor experiments on Pythia.

Author: Carson Molder
"""

from itertools import combinations, product
import os
from typing import List, Optional, Tuple

from tqdm import tqdm

from exp_utils import defaults, pc_trace
from exp_utils.run import pref_degree_knobs
from exp_utils.file import ChampsimTraceFile, ChampsimTraceDirectory

condor_template = 'experiments/exp_utils/condor_template.txt'
script_template = 'experiments/exp_utils/script_template.txt'


class Run():
    """Defines an individual run of a Condor sweep."""

    def __init__(self, cfg, trace: ChampsimTraceFile,
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
        """Initialize a run with the given parameters.
        """
        # Sanity checks
        assert len(l1d_prefetcher) == len(l1d_prefetcher_degree)
        assert len(l2_prefetcher) == len(l2_prefetcher_degree)
        assert len(llc_prefetcher) == len(llc_prefetcher_degree)

        self.trace = trace
        self.cfg = cfg

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

    def run_name(self) -> str:
        """Generate a run name that summarizes the run's parameters."""
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
        """Return a run name that summarizes the run's parameters."""
        return self.run_name()

    def mix_config_files(self, base_config_path: str) -> str:
        """Mix this run's parameters with a base config ini file.
        
        Parameters:
            base_config_path: Path to the base config ini file.
        
        Returns:
            mixed_config: String representing the mixed config (not 
               saved to file).

        TODO: Support additional knobs (e.g. prefetcher type, degree)
        """
        with open(base_config_path, 'r') as f:
            config = f.readlines()

        knobs = {}
        # Simulator seed
        knobs['champsim_seed'] = self.seed
        knobs['scooby_seed'] = self.seed
        # Pythia dynamic level
        if self.pythia_level_threshold is not None:
            knobs['scooby_enable_dyn_level'] = True
            knobs['scooby_dyn_level_threshold'] = self.pythia_level_threshold
        else:
            knobs['scooby_enable_dyn_level'] = False
        # Pythia low-confidence prefetch table
        if 'pythia' in self.cfg and self.cfg.pythia.separate_lowconf_pt:
            knobs['scooby_separate_lowconf_pt'] = True
            knobs['scooby_lowconf_pt_size'] = self.cfg.pythia.lowconf_pt_size
        # Pythia features
        if self.pythia_features is not None:
            knobs['le_featurewise_active_features'] = (
                ','.join([str(f) for f in self.pythia_features]))
            knobs['le_featurewise_enable_tiling_offset'] = (
                ','.join(["1" for _ in self.pythia_features])
            )
        # Pythia pooling type (1 = sum, 2 = max)
        knobs['le_featurewise_pooling_type'] = (1 
            if 'pythia' in self.cfg and self.cfg.pythia.pooling == 'sum' else 2)
        # Pythia dynamic degree
        knobs['scooby_enable_dyn_degree'] = (
            'pythia' in self.cfg and self.cfg.pythia.dyn_degree)
        # Pythia parameters
        if 'pythia' in self.cfg:
            knobs['scooby_alpha'] = self.cfg.pythia.alpha
            knobs['scooby_gamma'] = self.cfg.pythia.gamma
            knobs['scooby_epsilon'] = self.cfg.pythia.epsilon
            knobs['scooby_policy'] = self.cfg.pythia.policy
            knobs['scooby_learning_type'] = self.cfg.pythia.learning_type
            knobs['scooby_pt_size'] = self.cfg.pythia.pt_size

        clean_value = lambda v : str(v).lower() if isinstance(v, bool) or v is None else str(v)

        # Replace matched knobs in the config
        for i, line in enumerate(config):
            for k, v in knobs.items():
                if k in line:
                    config[i] = f'{k} = {clean_value(v)}\n'

        
        # Add unmatched knobs to the config.
        for k, v in knobs.items():
            if v != None: # Unmatched
                config.append(f'{k} = {clean_value(v)}\n')

        return ''.join(config)


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

        Returns:
            candidates: A list of valid degree options for the given
               prefetcher / prefetcher hybrid.
        """

        candidates = [(0,) for _ in range(len(prefetcher))]
        for i, pref in enumerate(prefetcher):
            # Filter out and don't sweep degree for the following:
            # 1) No prefetcher
            # 2) Prefetchers with dynamic degree schemes enabled
            # 3) Prefetchers that lack degree knobs
            if ('no' in pref
                or 'bingo' in pref and self.cfg.bingo.dyn_degree is True
                or 'scooby' in pref and self.cfg.pythia.dyn_degree is True
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
                            self.cfg, trace,
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


def generate_condor_config(out, dry_run, memory=0, **params):
    """Generate a configuration that Condor will use to submit the job."""
    with open(condor_template, 'r') as f:
        cfg = f.read()

    params['memory'] = memory
    cfg = cfg.format(**params)

    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=f)


def generate_condor_script(run: Run, cfg, script_out_path: str, 
                           config_path: str,
                           results_dir: str,
                           pc_trace_file: Optional[str] = None,
                           pc_trace_credit: bool = False,
                           pc_trace_invoke_all = False,
                           pref_trace_file: Optional[str] = None,
                           track_pc: bool = False,
                           track_addr: bool = False,
                           track_pref: bool = False,
                           dry_run: bool = False):
    """Generate a script that the Condor run will execute to simulate."""
    with open(script_template, 'r') as f:
        script = f.read()

    # Add required parameters
    script = script.format(
      champsim_dir = cfg.paths.champsim_dir,
      trace_file = run.trace.path,
      cores = 1,  # TODO: Support multicore
      config = config_path,
      l1d_pref = ' '.join(run.l1d_prefetcher),
      l2_pref = ' '.join(run.l2_prefetcher),
      llc_pref = ' '.join(run.llc_prefetcher),
      llc_repl = cfg.llc.repl,
      llc_sets = cfg.llc.sets,
      results_dir = results_dir,
      num_instructions = cfg.champsim.sim_instructions,
      warmup_instructions = cfg.champsim.warmup_instructions,
    )

    # TODO: Support L1D prefetcher degree
    # TODO: Move more of these knobs into the config.ini file.
    #cfg += f' \\\n    --l1d-pref-degrees ' + ' '.join([str(d) for d in run.l1d_prefetcher_degree])
    script += f' \\\n    --l2c-pref-degrees ' + ' '.join([str(d) for d in run.l2_prefetcher_degree])
    script += f' \\\n    --llc-pref-degrees ' + ' '.join([str(d) for d in run.llc_prefetcher_degree])
    script += f' \\\n    --run-name {run.run_name()}'
    if pc_trace_file is not None:
        script += f' \\\n    --pc-trace-llc {pc_trace_file}'
    if pc_trace_credit:
        script += f' \\\n    --pc-trace-credit'
    if pc_trace_invoke_all:
        script += f' \\\n    --pc-trace-invoke-all'
    if pref_trace_file:
        script += f' \\\n    --pref-trace-llc {pref_trace_file}'
    if track_pc:
        script += f' \\\n    --track-pc'
    if track_addr:
        script += f' \\\n    --track-addr'
    if track_pref:
        script += f' \\\n    --track-pref'

    if not dry_run:
        with open(script_out_path, 'w') as f:
            print(script, file=f)
        os.chmod(script_out_path, 0o755)  # Make script executable


def generate_condor_list(out, condor_paths):
    with open(out, 'w') as f:
        for path in condor_paths:
            print(path, file=f)


def build_run(cfg, run: Run,
              dry_run: bool = False,
              verbose: bool = False) -> str:
    """Build a single run and its necessary files. Return
    the path to the saved condor file.

    Parameters:
        cfg: Config dictionary describing sweep parameters.
        run: Run object describing run parameters.  
        dry_run: If true, does not save anything.
        verbose: If true, print more information.

    Return:
        condor_file: Path to Condor file
    """
    run_name = run.run_name()

    # Setup initial output directories/files per experiment
    log_file_base = os.path.join(cfg.paths.exp_dir, 'logs', run_name)
    condor_file = os.path.join(cfg.paths.exp_dir, 'condor',
                               f'{run_name}.condor')
    config_file = os.path.join(cfg.paths.exp_dir, 'configs', f'{run_name}.ini')
    script_file = os.path.join(cfg.paths.exp_dir, 'scripts', f'{run_name}.sh')
    results_dir = os.path.join(cfg.paths.exp_dir, 'champsim_results')

    if verbose:
        print(f'\nFiles for {run_name}:')
        print(f'    output log  : {log_file_base}.OUT')
        print(f'    error log   : {log_file_base}.ERR')
        print(f'    condor      : {config_file}')
        print(f'    config      : {condor_file}')
        print(f'    script      : {script_file}')
        print(f'    results dir : {results_dir}')

    # Create directories
    if not dry_run:
        os.makedirs(os.path.join(cfg.paths.exp_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.exp_dir, 'condor'), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.exp_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.exp_dir, 'scripts'), exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    # Build Condor file
    generate_condor_config(
        condor_file,
        dry_run,
        memory=0,
        condor_user=cfg.condor.user,
        condor_group=cfg.condor.group,
        condor_project=cfg.condor.project,
        condor_desc=cfg.condor.description,
        err_file=log_file_base + '.ERR',
        out_file=log_file_base + '.OUT',
        init_dir=cfg.paths.champsim_dir,
        exe=script_file,
    )

    # Build config file
    # TODO: Add parameter for base config to sweep config yml
    config = run.mix_config_files(
        defaults.default_knobs_file 
    )
    with open(config_file, 'w') as f:
        f.write(config)

    if verbose:
        print(f'ChampSim simulation parameters for {run_name}:')
        print(f'    targets        : {" ".join(run.llc_prefetcher)}')
        print(f'    experiment dir : {cfg.paths.exp_dir}')
        print(f'    champsim path  : {cfg.paths.champsim_dir}')
        print(f'    results dir    : {results_dir}')
        print(f'    # cores        : {1}')
        print(f'    # instructions : {cfg.champsim.sim_instructions} million')
        print(
            f'    # warmup insts : {cfg.champsim.warmup_instructions} million')

    # TODO: Cleanup the PC/Prefetch trace checks below.
    # Add PC trace path, if we are running the pc_trace prefetcher
    if run.llc_prefetcher == ('pc_trace', ):
        pc_trace_file = os.path.join(
            cfg.paths.pc_trace_dir,
            pc_trace.get_pc_trace_file(run.trace.full_trace,
                                       cfg.pc_trace.metric, level='llc'))
        pc_trace_credit = cfg.pc_trace.credit
        pc_trace_invoke_all = cfg.pc_trace.invoke_all

        if verbose:
            print(f'    pc_trace file  : {pc_trace_file}')
    else:
        pc_trace_file = None
        pc_trace_invoke_all = False
        pc_trace_credit = False

    # Add prefetch trace path, if we are running the from_file prefetcher
    # NOTE: Running from_file for the Prefetcher zoo defaults to the relevant offline PC trace.
    if run.llc_prefetcher == ('from_file',):
        pref_trace_file = os.path.join(
            cfg.paths.pref_trace_dir,
            pc_trace.get_pref_trace_file(run.trace.full_trace,
                                         cfg.pref_trace.metric, level='llc'))

        if verbose:
            print(f'    pref_trace file  : {pref_trace_file}')
    else:
        pref_trace_file = None

    # Generate Condor script
    generate_condor_script(
        run, cfg, script_file, config_file, results_dir,
        pc_trace_file=pc_trace_file,
        pc_trace_credit=pc_trace_credit,
        pc_trace_invoke_all=pc_trace_invoke_all,
        pref_trace_file=pref_trace_file,
        track_pc=cfg.champsim.track_pc_stats,
        track_addr=cfg.champsim.track_addr_stats,
        track_pref=cfg.champsim.track_pref,
        dry_run=dry_run
    )

    # Add condor file to the list
    return condor_file


def build_sweep(cfg, dry_run=False, verbose=False):
    """Build an evaluation sweep, for prefetcher_zoo.py
    """
    # Assertion checks
    if cfg.llc.pref_candidates == ('pc_trace', ):
        assert 'pc_trace_dir' in cfg.paths.keys(), (
            'Must add a PC trace directory to paths.pc_trace_dir '
            'if sweeping on pc_trace')
        assert cfg.pc_trace.metric in pc_trace.metrics, (
            f'PC trace metric {cfg.pc_trace.metric} '
            f'not in options {pc_trace.metrics}')

    # Get best degrees (if provided)
    # degrees_df = (pd.read_csv(cfg.paths.degree_csv) if 'degree_csv' in cfg.paths else None)
    print('Generating runs...')

    condor_paths = []
    run_generator = RunGenerator(cfg)
    for run in tqdm(run_generator, dynamic_ncols=True, unit='run'):
        condor_path = build_run(cfg, run,
                                dry_run=dry_run, verbose=verbose)
        condor_paths.append(condor_path)

    print(f'Generated {len(condor_paths)} runs')

    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    if not dry_run:
        condor_out_path = os.path.join(cfg.paths.exp_dir,
                                       'condor_configs_champsim.txt')
        print(f'Saving condor configs to {condor_out_path}...')
        generate_condor_list(condor_out_path, condor_paths)
