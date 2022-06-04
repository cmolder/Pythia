"""
Utility functions for setting up Condor experiments on Pythia.

Author: Carson Molder
"""

import pandas as pd
import os
import glob
from tqdm import tqdm
from itertools import combinations, product
from exp_utils import defaults, run, pc_trace
from exp_utils.file import ChampsimTraceFile

condor_template = 'experiments/exp_utils/condor_template.txt'
script_template = 'experiments/exp_utils/script_template.txt'


def generate_condor_config(out, dry_run, memory=0, **params):
    """Generate a configuration that Condor will use to submit the job."""
    with open(condor_template, 'r') as f:
        cfg = f.read()

    params['memory'] = memory
    cfg = cfg.format(**params)

    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=f)


def generate_condor_script(out, dry_run, **params):
    """Generate a script that the Condor run will execute to simulate."""
    with open(script_template, 'r') as f:
        cfg = f.read()

    # Add required parameters
    cfg = cfg.format(**params)

    # Add optional parameters
    if ('l2c_pref_degrees' in params.keys() 
        and params['l2c_pref_degrees'] is not None):
        cfg += f' \\\n    --l2c-pref-degrees {params["l2c_pref_degrees"]}'
    if ('llc_pref_degrees' in params.keys() 
        and params['llc_pref_degrees'] is not None):
        cfg += f' \\\n    --llc-pref-degrees {params["llc_pref_degrees"]}'
    if ('pc_trace_file' in params.keys() 
        and params['pc_trace_file'] is not None):
        cfg += f' \\\n    --pc-trace-llc {params["pc_trace_file"]}'
    if 'track_pc' in params.keys() and params['track_pc'] is True:
        cfg += f' \\\n    --track-pc'
    if 'track_addr' in params.keys() and params['track_addr'] is True:
        cfg += f' \\\n    --track-addr'
    if 'run_name' in params.keys() and params['run_name'] is not None:
        cfg += f' \\\n    --run-name {params["run_name"]}'
    if 'extra_knobs' in params.keys() and params['extra_knobs'] is not None:
        cfg += f' \\\n    --extra-knobs "\'{params["extra_knobs"]}\'"'

    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=f)
        os.chmod(out, 0o755)  # Make script executable


def generate_condor_list(out, condor_paths):
    with open(out, 'w') as f:
        for path in condor_paths:
            print(path, file=f)


def generate_run_name(trace_path,
                      llc_sets,
                      branch_pred,
                      llc_repl,
                      l1d_pref=[],
                      l2c_pref=[],
                      l2c_pref_degrees=[],
                      llc_pref=[],
                      llc_pref_degrees=[],
                      extra_suffix=None):
    """Generate a unique run name, given the trace, and prefetchers+degrees."""
    trace_name = '.'.join(os.path.basename(trace_path).split('.')[:-1])

    if len(l2c_pref_degrees) == len(l2c_pref):
        l2c_pref_degrees_suffix = ",".join([str(d) for d in l2c_pref_degrees])
    else:
        l2c_pref_degrees_suffix = ",".join(['na' for d in l2c_pref])

    if len(llc_pref_degrees) == len(llc_pref):
        llc_pref_degrees_suffix = ",".join([str(d) for d in llc_pref_degrees])
    else:
        llc_pref_degrees_suffix = ",".join(['na' for d in llc_pref])

    # More verbose
    # return '-'.join((
    #     trace_name,
    #     f'bp_{branch_pred}',
    #     f'l1c_pref_{",".join(l1d_pref)}',
    #     f'l2c_pref_{",".join(l2c_pref)}',
    #     f'l2c_pref_degrees_{l2c_pref_degrees_suffix}',
    #     f'llc_pref_{",".join(llc_pref)}',
    #     f'llc_pref_degrees_{llc_pref_degrees_suffix}',
    #     f'llc_sets_{llc_sets}',
    #     f'llc_repl_{llc_repl}'
    # ))

    # Matches binary
    out = '-'.join((trace_name, branch_pred, ','.join(l1d_pref),
                    ','.join(l2c_pref) + '_' + l2c_pref_degrees_suffix,
                    ','.join(llc_pref) + '_' + llc_pref_degrees_suffix,
                    llc_repl, f'{llc_sets}llc_sets'))
    if extra_suffix is not None:
        out += '-' + extra_suffix
    return out


def build_run(cfg,
              tr_path,
              l1d_pref=['no'],
              l2c_pref=['no'],
              l2c_pref_degrees=[],
              llc_pref=['no'],
              llc_pref_degrees=[],
              extra_suffix=None,
              extra_knobs=None,
              dry_run=False,
              verbose=False):
    """Build a single run and its necessary files. Return
    the path to the saved condor file.
    
    Parameters:
        tr_path: string
            Path to trace
            
        llc_pref: List[string]
            List of prefetchers that are in multi.llc_pref 
            (or "no", "pc_trace") for knob
            
        llc_num_sets: int
            Number of LLC sets

        exp_dir: string
            Directory of experiment.
            
        champsim_dir: string
            Directory of ChampSim / Pythia files. (default: $PYTHIA_HOME)
            
        num_instructions: int
            Number of instructions to simulate (in millions)
            
        warmup_instructions: int
            Number of instructions to warm up, before simulating (in millions)
            
        dry_run: string (optional)
            If raised, does not save anything.
            
        verbose: string (optional)
            If raised, print more information.
    
    Return:
        condor_file: string
            Path to Condor file
    """
    #tr_path = tr_path.replace('.txt', '.trace')
    run_name = generate_run_name(tr_path,
                                 cfg.llc.sets,
                                 cfg.champsim.branch_pred,
                                 cfg.llc.repl,
                                 l1d_pref=l1d_pref,
                                 l2c_pref=l2c_pref,
                                 l2c_pref_degrees=l2c_pref_degrees,
                                 llc_pref=llc_pref,
                                 llc_pref_degrees=llc_pref_degrees,
                                 extra_suffix=extra_suffix)

    # Setup initial output directories/files per experiment
    log_file_base = os.path.join(cfg.paths.exp_dir, 'logs', run_name)
    condor_file = os.path.join(cfg.paths.exp_dir, 'condor',
                               f'{run_name}.condor')
    script_file = os.path.join(cfg.paths.exp_dir, 'scripts', f'{run_name}.sh')
    results_dir = os.path.join(cfg.paths.exp_dir, 'champsim_results')

    if verbose:
        print(f'\nFiles for {run_name}:')
        print(f'    output log  : {log_file_base}.OUT')
        print(f'    error log   : {log_file_base}.ERR')
        print(f'    condor      : {condor_file}')
        print(f'    script      : {script_file}')
        print(f'    results dir : {results_dir}')

    # Create directories
    if not dry_run:
        os.makedirs(os.path.join(cfg.paths.exp_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.exp_dir, 'condor'), exist_ok=True)
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

    if verbose:
        print(f'ChampSim simulation parameters for {run_name}:')
        print(f'    targets        : {" ".join(llc_pref)}')
        print(f'    experiment dir : {cfg.paths.exp_dir}')
        print(f'    champsim path  : {cfg.paths.champsim_dir}')
        print(f'    results dir    : {results_dir}')
        print(f'    # cores        : {1}')
        print(f'    # instructions : {cfg.champsim.sim_instructions} million')
        print(
            f'    # warmup insts : {cfg.champsim.warmup_instructions} million')

    # Add PC trace path, if we are running the pc_trace prefetcher.
    if llc_pref == ('pc_trace', ):
        full_trace = ChampsimTraceFile(tr_path).full_trace
        pc_trace_file = os.path.join(
            cfg.paths.pc_trace_dir,
            pc_trace.get_pc_trace_file(full_trace,
                                       cfg.pc_trace_metric,
                                       level='llc'))

        if verbose:
            print(f'    pc_trace file  : {pc_trace_file}')
    else:
        pc_trace_file = None

    # Generate Condor script
    generate_condor_script(
        script_file,
        dry_run,
        champsim_dir=cfg.paths.champsim_dir,
        trace_file=tr_path,
        cores=1,
        l1d_pref=' '.join(l1d_pref),
        l2c_pref=' '.join(l2c_pref),
        l2c_pref_degrees=' '.join([str(d) for d in l2c_pref_degrees])
        if len(l2c_pref_degrees) > 0 else None,
        llc_pref=' '.join(llc_pref),
        llc_pref_degrees=' '.join([str(d) for d in llc_pref_degrees])
        if len(llc_pref_degrees) > 0 else None,
        llc_repl=cfg.llc.repl,
        llc_sets=cfg.llc.sets,
        run_name=run_name,
        pc_trace_file=pc_trace_file,
        results_dir=results_dir,
        extra_knobs=extra_knobs,
        warmup_instructions=cfg.champsim.warmup_instructions,
        num_instructions=cfg.champsim.sim_instructions,
        track_pc=cfg.champsim.track_pc_pref,
        track_addr=cfg.champsim.track_addr_pref,
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
        assert cfg.pc_trace_metric in pc_trace.metrics, (
            f'PC trace metric {cfg.pc_trace_metric} '
            f'not in options {pc_trace.metrics}')

    # Get best degrees (if provided)
    degrees = (pd.read_csv(cfg.paths.degree_csv) 
               if 'degree_csv' in cfg.paths else None)

    condor_paths = []
    paths = glob.glob(os.path.join(cfg.paths.trace_dir, '*.trace.*'))

    # Get all combinations of hybrids up to <max_hybrid>
    l1d_prefs = [
        p for h in range(1, cfg.l1d.max_hybrid + 1)
        for p in combinations(cfg.l1d.pref_candidates, h)
    ] + [('no', )]
    l2c_prefs = [
        p for h in range(1, cfg.l2c.max_hybrid + 1)
        for p in combinations(cfg.l2c.pref_candidates, h)
    ] + [('no', )]
    llc_prefs = [
        p for h in range(1, cfg.llc.max_hybrid + 1)
        for p in combinations(cfg.llc.pref_candidates, h)
    ] + [('no', )]

    print('Generating runs...')
    with tqdm(dynamic_ncols=True, unit='run') as pbar:
        for path in paths:
            for l1p, l2p, llp in product(l1d_prefs, l2c_prefs, llc_prefs):

                trace_name = ChampsimTraceFile(path).full_trace

                if not isinstance(degrees, type(None)):
                    l2c_pref_degree = list(
                        eval(degrees[degrees.Trace == trace_name][str(
                            ('_'.join(l1p), '_'.join(l2p), '_'.join(llp)
                             ))].item())[0]) if l2p != ('no', ) else []
                    llc_pref_degree = list(
                        eval(degrees[degrees.Trace == trace_name][str(
                            ('_'.join(l1p), '_'.join(l2p), '_'.join(llp)
                             ))].item())[1]) if llp != ('no', ) else []
                else:
                    l2c_pref_degree, llc_pref_degree = [], []

                #print('[DEBUG]', trace_name, l1p, l2p, llp, l2c_pref_degree, llc_pref_degree)

                c_path = build_run(cfg,
                                   path,
                                   l1d_pref=l1p,
                                   l2c_pref=l2p,
                                   llc_pref=llp,
                                   l2c_pref_degrees=l2c_pref_degree,
                                   llc_pref_degrees=llc_pref_degree,
                                   dry_run=dry_run,
                                   verbose=verbose)
                condor_paths.append(c_path)
                pbar.update(1)

    print(f'Generated {len(condor_paths)} runs')

    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    if not dry_run:
        condor_out_path = os.path.join(cfg.paths.exp_dir,
                                       'condor_configs_champsim.txt')
        print(f'Saving condor configs to {condor_out_path}...')
        generate_condor_list(condor_out_path, condor_paths)


def _should_skip_degree_combination(l2c_pref, llc_pref, degs):
    """Helper function to skip redundant degree sweeps,
    on prefetchers that aren't tunable w.r.t degree.
    
    Check if any of the prefetchers does not have a degree knob,
    f so, don't sweep over that degree (default to 1, which
    gets ignored when run detects it's not a valid degree-tunable
    prefetcher)
    """
    prefs = (*l2c_pref, *llc_pref)
    #print('[DEBUG]', prefs, degs)

    for i, pref in enumerate(prefs):
        if pref not in run.pref_degree_knobs.keys() and degs[i] != 1:
            return True
    return False


def build_degree_sweep(cfg, dry_run=False, verbose=False):
    """Build a degree sweep, for prefetcher_degree_sweep.py
    """
    # Assertion checks
    assert 'pc_trace' not in cfg.llc.pref_candidates, (
        'Cannot tune pc_trace for degree')

    condor_paths = []
    paths = glob.glob(os.path.join(cfg.paths.trace_dir, '*.trace.*'))

    # Get all combinations of hybrids up to <max_hybrid>
    l1d_prefs = [
        p for h in range(1, cfg.l1d.max_hybrid + 1)
        for p in combinations(cfg.l1d.pref_candidates, h)
    ] + [('no', )]
    l2c_prefs = [
        p for h in range(1, cfg.l2c.max_hybrid + 1)
        for p in combinations(cfg.l2c.pref_candidates, h)
    ] + [('no', )]
    llc_prefs = [
        p for h in range(1, cfg.llc.max_hybrid + 1)
        for p in combinations(cfg.llc.pref_candidates, h)
    ] + [('no', )]

    print('Generating runs...')
    with tqdm(dynamic_ncols=True, unit='run') as pbar:
        for path in paths:
            for l1p, l2p, llp in product(l1d_prefs, l2c_prefs, llc_prefs):
                for d in product(
                        *[list(range(1, cfg.l2c.max_degree + 1))] * len(l2p),
                        *[list(range(1, cfg.llc.max_degree + 1))] * len(llp)):

                    if _should_skip_degree_combination(l2p, llp, d):
                        continue

                    l2d, lld = d[:len(l2p)], d[len(l2p):]

                    #print('[DEBUG]', path, l1p, l2p, llp, l2d, lld)
                    c_path = build_run(cfg,
                                       path,
                                       l1d_pref=l1p,
                                       l2c_pref=l2p,
                                       l2c_pref_degrees=l2d,
                                       llc_pref=llp,
                                       llc_pref_degrees=lld,
                                       dry_run=dry_run,
                                       verbose=verbose)

                    condor_paths.append(c_path)
                    pbar.update(1)

    print(f'Generated {len(condor_paths)} runs')

    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    if not dry_run:
        condor_out_path = os.path.join(cfg.paths.exp_dir,
                                       'condor_configs_champsim.txt')
        print(f'Saving condor configs to {condor_out_path}...')
        generate_condor_list(condor_out_path, condor_paths)


def get_extra_knobs_pythia_level(cfg,
                                 seed=None,
                                 level_threshold=None,
                                 features=None):
    extra_knobs = ''

    if level_threshold is not None:
        extra_knobs += (' --scooby_enable_dyn_level=true'
                        f' --scooby_dyn_level_threshold={level_threshold}')
    else:
        extra_knobs += f' --scooby_enable_dyn_level=false'

    if seed is not None:
        extra_knobs += f' --champsim_seed={seed} --scooby_seed={seed}'

    if cfg.pythia.scooby_separate_lowconf_pt is True:
        extra_knobs += f' --scooby_separate_lowconf_pt=true'
    else:
        extra_knobs += f' --scooby_separate_lowconf_pt=false'

    if features is not None:
        extra_knobs += (' --le_featurewise_active_features='
                        f'{",".join([str(f) for f in features])}')
        extra_knobs += (' --le_featurewise_enable_tiling_offset='
                        f'{"1," * len(features)}')

    extra_knobs += f' --scooby_alpha={cfg.pythia.scooby_alpha}'
    extra_knobs += f' --scooby_gamma={cfg.pythia.scooby_gamma}'
    extra_knobs += f' --scooby_epsilon={cfg.pythia.scooby_epsilon}'
    extra_knobs += f' --scooby_policy={cfg.pythia.scooby_policy}'
    extra_knobs += f' --scooby_learning_type={cfg.pythia.scooby_learning_type}'
    extra_knobs += f' --scooby_pt_size={cfg.pythia.scooby_pt_size}'
    extra_knobs += (' --scooby_lowconf_pt_size='
                    f'{cfg.pythia.scooby_lowconf_pt_size}')

    return extra_knobs


def any_pythia(l1p, l2p, llp):
    """Return True if any of the prefetchers
    are Pythia (or Pythia-related).
    """
    for p in [l1p, l2p, llp]:
        if 'scooby' in p or 'scooby_double' in p:
            return True
    return False


def build_pythia_level_sweep(cfg, dry_run=False, verbose=False):
    """Build a level-aware Pythia sweep, for pythia_level.py
    """
    condor_paths = []
    paths = glob.glob(os.path.join(cfg.paths.trace_dir, '*.[g|x]z'))

    # Get all combinations of hybrids up to <max_hybrid>
    l1d_prefs = [
        p for h in range(1, cfg.l1d.max_hybrid + 1)
        for p in combinations(cfg.l1d.pref_candidates, h)
    ] + [('no', )]
    l2c_prefs = [
        p for h in range(1, cfg.l2c.max_hybrid + 1)
        for p in combinations(cfg.l2c.pref_candidates, h)
    ] + [('no', )]
    llc_prefs = [
        p for h in range(1, cfg.llc.max_hybrid + 1)
        for p in combinations(cfg.llc.pref_candidates, h)
    ] + [('no', )]

    print('Generating runs...')
    with tqdm(dynamic_ncols=True, unit='run') as pbar:
        for path in paths:
            for seed in cfg.champsim.seeds:
                for l1p, l2p, llp in product(l1d_prefs, l2c_prefs, llc_prefs):
                    for feats in cfg.pythia.scooby_features:
                        for thresh in cfg.pythia.scooby_dyn_level_threshold:

                            if all([p == ('no', )
                                    for p in (l1p, l2p, llp)]) or any([
                                        p == ('scooby_double', )
                                        for p in (l1p, l2p, llp)
                                    ]):
                                continue

                            c_path = build_run(
                                cfg,
                                path,
                                l1d_pref=l1p,
                                l2c_pref=l2p,
                                llc_pref=llp,
                                extra_knobs=get_extra_knobs_pythia_level(
                                    cfg,
                                    seed=seed,
                                    level_threshold=thresh,
                                    features=feats,
                                ),
                                extra_suffix=(
                                    f'threshold_{thresh}_'
                                    'features_'
                                    f'{",".join([str(f) for f in feats])}_'
                                    f'seed_{seed}'),
                                dry_run=dry_run,
                                verbose=verbose)

                            condor_paths.append(c_path)
                            pbar.update(1)

                        # Add runs for Double Pythia (extra actions for 
                        # LLC prefetches), Static Pythia
                        if any_pythia(l1p, l2p, llp):
                            c_path = build_run(
                                cfg,
                                path,
                                l1d_pref=l1p,
                                l2c_pref=l2p,
                                llc_pref=llp,
                                extra_knobs=get_extra_knobs_pythia_level(
                                    cfg,
                                    seed=seed,
                                    level_threshold=None,
                                    features=feats),
                                extra_suffix=(
                                    'features_'
                                    f'{",".join([str(f) for f in feats])}_'
                                    f'seed_{seed}'),
                                dry_run=dry_run,
                                verbose=verbose)

                            condor_paths.append(c_path)
                            pbar.update(1)

                    # Add runs for other prefetchers / no prefetcher
                    if not any_pythia(l1p, l2p, llp):
                        c_path = build_run(
                            cfg,
                            path,
                            l1d_pref=l1p,
                            l2c_pref=l2p,
                            llc_pref=llp,
                            extra_knobs=get_extra_knobs_pythia_level(
                                cfg,
                                seed=seed,
                                level_threshold=None,
                                features=None),
                            extra_suffix=f'seed_{seed}',
                            dry_run=dry_run,
                            verbose=verbose)

                        condor_paths.append(c_path)
                        pbar.update(1)

    print(f'Generated {len(condor_paths)} runs')

    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    if not dry_run:
        condor_out_path = os.path.join(cfg.paths.exp_dir,
                                       'condor_configs_champsim.txt')
        print(f'Saving condor configs to {condor_out_path}...')
        generate_condor_list(condor_out_path, condor_paths)
