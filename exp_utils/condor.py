"""
Utility functions for setting up Condor experiments on Pythia.

Author: Carson Molder
"""

import os
import glob
from itertools import combinations
from exp_utils import defaults

def generate_condor_config(out, dry_run, memory=0, **params):
    """Generate a configuration that Condor will use to submit the job."""
    with open('exp_utils/condor_template.txt', 'r') as f:
        cfg = f.read()
    
    params['memory'] = memory
    cfg = cfg.format(**params)
        
    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=f)

            
def generate_condor_script(out, dry_run, **params):
    """Generate a script that the Condor run will execute to simulate."""
    with open('exp_utils/script_template.txt', 'r') as f:
        cfg = f.read()
        
    cfg = cfg.format(**params)
    
    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=f)
        os.chmod(out, 0o777) # Make script executable
        

def generate_condor_list(out, condor_paths):
    with open(out, 'w') as f:
        for path in condor_paths:
            print(path, file=f)


def generate_run_name(trace_path, llc_num_sets, llc_prefetchers):
    """Generate a unique run name, given the trace, LLC number of sets, and LLC prefetchers."""
    trace_name = os.path.basename(trace_path).split('.')[0]
    
    return '-'.join((
        trace_name, 
        f'llc_pref_{",".join(llc_prefetchers)}', 
        f'llc_sets_{llc_num_sets}'
    ))
    


def build_run(tr_path, llc_prefetchers,
              llc_num_sets=defaults.default_llc_sets,
              exp_dir=defaults.default_exp_dir,
              champsim_dir=defaults.default_champsim_dir,
              num_instructions=defaults.default_sim_instructions,
              warmup_instructions=defaults.default_warmup_instructions,
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
    run_name = generate_run_name(tr_path, llc_num_sets, llc_prefetchers)
    
    # Setup initial output directories/files per experiment
    log_file_base = os.path.join(exp_dir, 'logs', run_name)
    condor_file = os.path.join(exp_dir, 'condor', f'{run_name}.condor')
    script_file = os.path.join(exp_dir, 'scripts', f'{run_name}.sh')
    results_dir = os.path.join(exp_dir, 'champsim_results')
    
    if verbose:
        print(f'\nFiles for {run_name}:')
        print(f'    output log  : {log_file_base}.OUT')
        print(f'    error log   : {log_file_base}.ERR')
        print(f'    condor      : {condor_file}')
        print(f'    script      : {script_file}')
        print(f'    results dir : {results_dir}')
        
    # Create directories
    if not dry_run:
        os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'condor'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'scripts'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, results_dir), exist_ok=True)
        
    # Build Condor file
    generate_condor_config(
        condor_file,
        dry_run,
        memory=0,
        err_file=log_file_base + '.ERR',
        out_file=log_file_base + '.OUT',
        init_dir=champsim_dir,
        exe=script_file,
    )
        
    if verbose:
        print(f'ChampSim simulation parameters for {run_name}:')
        print(f'    targets        : {" ".join(targets)}')
        print(f'    experiment dir : {exp_dir}')
        print(f'    champsim path  : {champsim_dir}')
        print(f'    results dir    : {results_dir}')
        print(f'    # cores        : {1}')
        print(f'    # instructions : {num_instructions} million')
        print(f'    # warmup insts : {warmup_instructions} million')
    
    generate_condor_script(
        script_file,
        dry_run,
        champsim_dir=champsim_dir,
        conda_source=defaults.default_conda_source,
        trace_file=tr_path,
        num_cores=1,
        num_sets=llc_num_sets,
        targets=' '.join(llc_prefetchers),
        results_dir=results_dir,
        num_instructions=num_instructions,
        warmup_instructions=warmup_instructions
    )

    # Add condor file to the list
    return condor_file

def build_sweep(trace_dir, llc_prefetchers,
                max_hybrid=defaults.default_max_hybrid,
                llc_num_sets=defaults.default_llc_sets,
                exp_dir=defaults.default_exp_dir,
                champsim_dir=defaults.default_champsim_dir,
                num_instructions=defaults.default_sim_instructions,
                warmup_instructions=defaults.default_warmup_instructions,
                dry_run=False, 
                verbose=False):
    
    condor_paths = []
    
    for path in glob.glob(os.path.join(trace_dir, '*.trace.*')):
        for num_hybrid in range(1, max_hybrid + 1):
            for prefs in combinations(llc_prefetchers, num_hybrid):
                c_path = build_run(
                    path, prefs,              
                    llc_num_sets=llc_num_sets,
                    exp_dir=exp_dir,
                    champsim_dir=champsim_dir,
                    num_instructions=num_instructions,
                    warmup_instructions=warmup_instructions,
                    dry_run=dry_run, 
                    verbose=verbose
                )
                condor_paths.append(c_path)
        
        # Build no prefetcher baseline
        c_path = build_run(
            path, ['no'],              
            llc_num_sets=llc_num_sets,
            exp_dir=exp_dir,
            champsim_dir=champsim_dir,
            num_instructions=num_instructions,
            warmup_instructions=warmup_instructions,
            dry_run=dry_run, 
            verbose=verbose
        )
        condor_paths.append(c_path)
        
    # Write condor paths to <exp_dir>/condor_configs_champsim.txt
    generate_condor_list(
        os.path.join(exp_dir, 'condor_configs_champsim.txt'),
        condor_paths
    )
    
    print(f'Generated {len(condor_paths)} runs')