"""
Utility functions for setting up Condor experiments on Pythia.

Author: Carson Molder
"""

import os
import glob
from tqdm import tqdm
from itertools import combinations, product
from exp_utils import defaults, run, pc_trace, evaluate

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
    if 'llc_pref_degrees' in params.keys() and params['llc_pref_degrees'] is not None:
        cfg += f'\\\n    --llc-pref-degrees {params["llc_pref_degrees"]}'
    if 'pc_trace_file' in params.keys() and params['pc_trace_file'] is not None:
        cfg += f'\\\n    --pc-trace-llc {params["pc_trace_file"]}'
    
    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=f)
        os.chmod(out, 0o777) # Make script executable
        

def generate_condor_list(out, condor_paths):
    with open(out, 'w') as f:
        for path in condor_paths:
            print(path, file=f)


def generate_run_name(trace_path, llc_num_sets, llc_prefetchers, llc_pref_degrees=[]):
    """Generate a unique run name, given the trace, LLC number of sets, and LLC prefetchers."""
    trace_name = os.path.basename(trace_path).split('.')[0]
    
    if len(llc_pref_degrees) == len(llc_prefetchers):
        llc_pref_degrees_suffix = ",".join([str(d) for d in llc_pref_degrees])
    else:
        llc_pref_degrees_suffix = ",".join(['na' for d in llc_prefetchers])
    
    return '-'.join((
        trace_name, 
        f'llc_pref_{",".join(llc_prefetchers)}', 
        f'llc_pref_degrees_{llc_pref_degrees_suffix}',
        f'llc_sets_{llc_num_sets}'
    ))
    


def build_run(cfg, tr_path, llc_prefetchers,
              llc_pref_degrees=[],
              dry_run=False, 
              verbose=False):
    """Build a single run and its necessary files. Return
    the path to the saved condor file.
    
    Parameters:
        tr_path: string
            Path to trace
            
        llc_prefetchers: List[string]
            List of prefetchers that are in multi.llc_pref, for knob
            
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
    run_name = generate_run_name(tr_path, cfg.llc.sets, llc_prefetchers, llc_pref_degrees)
    
    # Setup initial output directories/files per experiment
    log_file_base = os.path.join(cfg.paths.exp_dir, 'logs', run_name)
    condor_file = os.path.join(cfg.paths.exp_dir, 'condor', f'{run_name}.condor')
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
        condor_file, dry_run,
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
        print(f'    targets        : {" ".join(llc_prefetchers)}')
        print(f'    experiment dir : {cfg.paths.exp_dir}')
        print(f'    champsim path  : {cfg.paths.champsim_dir}')
        print(f'    results dir    : {results_dir}')
        print(f'    # cores        : {1}')
        print(f'    # instructions : {cfg.champsim.sim_instructions} million')
        print(f'    # warmup insts : {cfg.champsim.warmup_instructions} million')
        
    # Add PC trace path, if we are running the pc_trace prefetcher.
    if llc_prefetchers == ('pc_trace',):
        full_trace = evaluate.get_full_trace_from_path(os.path.basename(tr_path).split('.')[0])
        pc_trace_file = os.path.join(
            cfg.paths.pc_trace_dir, 
            pc_trace.get_pc_trace_file(full_trace, cfg.pc_trace_metric, level='llc')
        )
        
        if verbose:
            print(f'    pc_trace file  : {pc_trace_file}' )    
    else:
        pc_trace_file = None
    
    # Generate Condor script
    generate_condor_script(
        script_file,
        dry_run,
        champsim_dir=cfg.paths.champsim_dir,
        #conda_source=cfg.conda.source,
        #conda_env=cfg.conda.env,
        trace_file=tr_path,
        num_cores=1,
        num_sets=cfg.llc.sets,
        llc_prefetchers=' '.join(llc_prefetchers),
        llc_pref_degrees=' '.join([str(d) for d in llc_pref_degrees]) if len(llc_pref_degrees) > 0 else None,
        pc_trace_file=pc_trace_file,
        results_dir=results_dir,
        warmup_instructions=cfg.champsim.warmup_instructions,
        num_instructions=cfg.champsim.sim_instructions,
    )

    # Add condor file to the list
    return condor_file

def build_sweep(cfg, dry_run=False, verbose=False):
    """Build an evaluation sweep, for prefetcher_zoo.py
    """  
    # Assertion checks
    if cfg.llc.pref_candidates == ('pc_trace',):
        assert 'pc_trace_dir' in cfg.paths.keys(), 'Must add a PC trace directory to paths.pc_trace_dir if sweeping on pc_trace'
        assert cfg.pc_trace_metric in pc_trace.metrics, f'PC trace metric {cfg.pc_trace_metric} not in options {pc_trace.metrics}'
    
    condor_paths = []
    paths = glob.glob(os.path.join(cfg.paths.trace_dir, '*.trace.*'))
    
    print('Generating runs...')
    with tqdm(dynamic_ncols=True, unit='run') as pbar:
        for path in paths:
            for num_hybrid in range(1, cfg.llc.max_hybrid + 1):
                for prefs in combinations(cfg.llc.pref_candidates, num_hybrid):
                    c_path = build_run(
                        cfg, path, prefs,              
                        dry_run=dry_run, 
                        verbose=verbose
                    )
                    condor_paths.append(c_path)
                    pbar.update(1)
        
            # Build no prefetcher baseline
            c_path = build_run(
                cfg, path, ['no'],              
                dry_run=dry_run, 
                verbose=verbose
            )
            condor_paths.append(c_path)
            pbar.update(1)
        
    print(f'Generated {len(condor_paths)} runs')
        
    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    if not dry_run:
        condor_out_path = os.path.join(cfg.paths.exp_dir, 'condor_configs_champsim.txt')
        print(f'Saving condor configs to {condor_out_path}...')
        generate_condor_list(condor_out_path, condor_paths)
    
    
    
def _should_skip_degree_combination(prefs, degs):
    """Helper function to skip redundant degree sweeps,
    on prefetchers that aren't tunable w.r.t degree.
    
    Check if any of the prefetchers does not have a degree knob,
    f so, don't sweep over that degree (default to 1, which
    gets ignored when run detects it's not a valid degree-tunable
    prefetcher)
    """
    for i, pref in enumerate(prefs):
        if pref not in run.pref_degree_knobs.keys() and degs[i] != 1:
            return True
    return False
    
def build_degree_sweep(cfg, dry_run=False, verbose=False):
    """Build a degree sweep, for prefetcher_degree_sweep.py
    """
    # Assertion checks
    assert 'pc_trace' not in cfg.llc.pref_candidates, 'Cannot tune pc_trace for degree'
    
    condor_paths = []
    paths = glob.glob(os.path.join(cfg.paths.trace_dir, '*.trace.*'))
    
    print('Generating runs...')
    with tqdm(dynamic_ncols=True, unit='run') as pbar:
        for path in paths:
            for num_hybrid in range(1, cfg.llc.max_hybrid + 1):
                for prefs in combinations(cfg.llc.pref_candidates, num_hybrid):
                    for degs in product(*[list(range(1, cfg.llc.max_degree + 1))]*num_hybrid):

                        # Skip degree combinations if one of the prefetchers is not
                        # degree-tunable (see run.pref_degree_knobs and 
                        # _should_skip_degree_combination)
                        if _should_skip_degree_combination(prefs, degs):
                            continue

                        #print(path, prefs, degs)
                        c_path = build_run(
                            cfg, path, prefs,
                            llc_pref_degrees=degs,
                            dry_run=dry_run, 
                            verbose=verbose
                        )
                        condor_paths.append(c_path)
                        pbar.update(1)
        
        # Build no prefetcher baseline
        c_path = build_run(
            cfg, path, ['no'],              
            dry_run=dry_run, 
            verbose=verbose
        )
        condor_paths.append(c_path)
        pbar.update(1)
        
    print(f'Generated {len(condor_paths)} runs')
    
    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    if not dry_run:
        condor_out_path = os.path.join(cfg.paths.exp_dir, 'condor_configs_champsim.txt')
        print(f'Saving condor configs to {condor_out_path}...')
        generate_condor_list(condor_out_path, condor_paths)
