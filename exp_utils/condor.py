"""
Utility functions for setting up Condor experiments on the Prefetching Zoo.

Author: Carson Molder
"""

import os

def generate_condor_config(out, dry_run, memory=0, **params):
    with open('exp_utils/condor_template.txt', 'r') as f:
        cfg = f.read()
    
    params = {
        memory=memory,
        **params
    }
    
    cfg = cfg.format(**params)
        
    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=out)

            
# TODO - Change keywords and template
def generate_condor_script(out, dry_run, **params):
    with open('exp_utils/script_template.txt', 'r') as f:
        cfg = f.read()
        
    cfg = cfg.format(**params)
    
    if not dry_run:
        with open(out, 'w') as f:
            print(cfg, file=out)
        os.chmod(out, 0o777) # Make script executable


def generate_run_name(trace_path, llc_num_sets, prefetchers):
    trace_name = os.path.basename(tr_path).split('_')[0]
    return trace_name + f'{'_'.join(prefetchers)} f'-{llc_num_sets}sets'
    


def build_run(tr_path, prefetchers,
              llc_num_sets=defaults.default_llc_sets,
              exp_dir=defaults.default_exp_dir,
              champsim_dir=defaults.default_champsim_dir,
              dry_run=False, verbose=False):
    """Build a single run and its necessary files. Return
    the path to the saved condor file.
    
    Parameters:
        tr_path: string
            Path to trace
            
        llc_num_sets: int
            Number of LLC sets
            
        prefetchers: List[string]
            List of prefetchers that are in multi.llc_pref, for knob
            
        exp_dir: string
            Directory of experiment.
            
        champsim_dir: string
            Directory of ChampSim / Pythia files. (default: $PYTHIA_HOME)
            
        dry_run: string (optional)
            If raised, does not save anything.
            
        verbose: string (optional)
            If raised, print more information.
    
    Return:
        condor_file: string
            Path to Condor file
    """
    tr_path = tr_path.replace('.txt', '.trace')
    run_name = generate_run_str(tr_path, llc_num_sets, prefetchers)
    
    
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
        
    # Determine number of warmup instructions, and total instructions
    num_warmup = int(round(num_inst * (defaults.default_warmup_pct/ 100))) # First (train+valid)% go to warmup.

    if verbose:
        print(f'ChampSim simulation parameters for {run_name}:')
        print(f'    targets        : {" ".join(targets)}')
        print(f'    experiment dir : {exp_dir}')
        print(f'    champsim path  : {champsim_dir}')
        print(f'    results dir    : {results_dir}')
        print(f'    # cores        : {1}')
        print(f'    # instructions : {defaults.default_instrs} million')
        print(f'    # warmup insts : {num_warmup} million')
    
    # TODO - Change keywords
    generate_condor_script(
        script_file,
        dry_run
        champsim_dir=champsim_dir,
        trace_file=tr_path,
        num_cores=1,
        num_sets=llc_num_sets,
        targets=' '.join(prefetchers),
        results_dir=results_dir,
        num_instructions=defaults.default_instrs,
    )

    # Add condor file to the list
    return condor_file
    