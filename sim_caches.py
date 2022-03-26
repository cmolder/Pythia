#!/usr/bin/env python3

"""
Build singlecore versions of ChampSim, and evaluate
LLC prefetchers.

Authors: Quang Duong and Carson Molder
"""

import argparse
import os
import sys
import shutil
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

from exp_utils import defaults, build, condor

# Example:
#   64 bytes per line
# x 16 ways per set (i.e. 1 set = 1 KB)
# x 2048 sets
# --------
#   2 MB cache size

# llc_num_sets = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768] # Number of sets
#              [256 KB, 512 KB, 1 MB, 2 MB, 4 MB, 8 MB, 16 MB, 32 MB]

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    build            Build ChampSim binaries
    run              Run ChampSim on specified traces
    condor_setup     Set up Condor prefetching zoo experiment
    eval             Parse and compute metrics on simulation results
    help             Display this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'build': '''usage: {prog} build <target> [-c / --cores <core-count-list>] [-s / --sets <llc-set-count-list>]

Description:
    {prog} build <prefetcher>
        Builds ChampSim binaries with an LLC <prefetcher>, where <prefetcher> is one of:

        no            No prefetcher
        multi         Runtime-configurable prefetcher (see multi.llc_pref for options)
        all           All of the above.
        
        Note: The replacement policy is SHiP for the LLC, and L1/L2 have no prefetcher.

Options:
    -c / --cores <core-count-list>
        Specifies a list of cores to build ChampSim variants. A single core
        version will always be built, but additional versions (e.g. 2-core / 4-core)
        can be listed here (e.g. using -c 2 4). The ChampSim script is tested up
        to 8 cores.
        Default: One core only

    -s / --sets <llc-set-count-list>
        Specifies a list of LLC set sizes to build ChampSim variants.
        Default: {default_llc_sets} sets only.

Notes:
    Barring updates to the GitHub repository, this will only need to be done once.
'''.format(
    prog=sys.argv[0], 
    default_llc_ways=defaults.default_llc_ways, 
    default_llc_sets=defaults.default_llc_sets
),
    
'condor_setup': '''usage: {prog} condor_setup <prefetchers> [-h / --hybrid <max-hybrid-count>]

Description:
    {prog} condor_setup <prefetchers>
        Sets up a Prefetching Zoo sweep for use on Condor. <prefetchers> is any set of LLC prefetchers
        defined in prefetcher/multi.llc_pref, separated by a space (" ").
        
Options:
    -d / --experiment-dir <experiment-dir>
        The directory to put the Condor scripts, results, etc.
        
        Default: {default_exp_dir}
        
    -t / --trace-dir <trace-dir>
        The directory where ChampSim traces will be found.
        
        Default {default_trace_dir}
    
    -h / --hybrid <max-hybrid-counts>
        Will build all combinations of LLC <prefetchers>, up to <max-hybrid-counts> running
        at the same time. For example, -h 2 will build configurations for all 2 hybrids, single prefetchers,
        and no prefetcher.
        
        Default: {default_max_hybrid}
        
    -s / --llc-sets <num-llc-sets>
        The number of LLC cache sets that ChampSim will be simulating. By default,
        {default_llc_sets} sets are used (if the binary is available).
        
    --warmup-instructions <warmup-instructions>
        Number of instructions to warmup the simulation for. Defaults to
        {default_warmup_instructions}M instructions

    --num-instructions <num-instructions>
        Number of instructions to run the simulation for. Defaults to
        {default_sim_instructions}M instructions
        
    -v / --verbose
        If passed, prints extra details about the experiment setup.
        
    --dry-run
        If passed, builds the experiment but writes nothing to <experiment-dir>.
'''.format(
    prog=sys.argv[0], 
    default_exp_dir=defaults.default_exp_dir,
    default_trace_dir=defaults.default_trace_dir,
    default_max_hybrid=defaults.default_max_hybrid,
    default_llc_sets=defaults.default_llc_sets,
    default_warmup_instructions=defaults.default_warmup_instructions,
    default_sim_instructions=defaults.default_sim_instructions
),
    

'run': '''usage: {prog} run <execution-traces> [-c / --cores <num-cores>] [-s / --sets <num-llc-sets>]
                            [-t / --targets <list-of-targets>] [--hawkeye-split <hawkeye-split>]
                            [--results-dir <results-dir>] [--num-instructions <num-instructions>] 
                            [--stat-printing-period <num-instructions>]

Description:
    {prog} run <execution-traces>
        Runs the base ChampSim binary on the specified execution trace(s). If using
        a multi-core setup, must provide <cores> traces.

Options:
    -c / --cores <num-cores>
        The number of cores that ChampSim will be simulating. Must provide a <cores>
        length list of execution traces to the script. By default, one core is used.

    -s / --sets <num-llc-sets>
        The number of LLC cache sets that ChampSim will be simulating. By default,
        {default_llc_sets} sets are used (if the binary is available).

    -t / --targets <list-of-targets>
        List of targets to run. By default, it will run all targets: {prefetcher_names}.
    
    --results-dir <results-dir>
        Specifies what directory to save the ChampSim results file in. This
        defaults to `{default_results_dir}`.
        
    --warmup-instructions <warmup-instructions>
        Number of instructions to warmup the simulation for. Defaults to
        {default_warmup_instructions}M instructions

    --num-instructions <num-instructions>
        Number of instructions to run the simulation for. Defaults to
        {default_sim_instructions}M instructions
'''.format(
    prog=sys.argv[0], 
    default_results_dir=defaults.default_results_dir,
    prefetcher_names=defaults.prefetcher_names,
    default_warmup_instructions=defaults.default_warmup_instructions,
    default_sim_instructions=defaults.default_sim_instructions,
    default_llc_sets=defaults.default_llc_sets,
),

'eval': '''usage: {prog} eval <results-dir> [--output-file <output-file>] [--norm-baseline <baseline>]

TODO - WORK IN PROGRESS FOR PREFETCHER ZOO! NOT READY FOR USE!

Description:
    {prog} eval
        Runs the evaluation procedure on the ChampSim result files found in the specified
        results directory and outputs a CSV at the specified output path.

Options:
    --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This defaults to
        `{default_output_file}`.
        
    --norm-baseline <baseline>
        If specified, results on ALL baselines will be normalized to this specific
        baseline, for both homogeneous-mix and single-core norms. If not specified,
        the normalization will be to the closest-related baseline (specified in
        the NormBaseline column).

Note:
    To get stats comparing performance to a no-prefetcher baseline, it is necessary
    to have run the base ChampSim binary on the same execution trace.

    Without the base data, relative performance data comparing MPKI and IPC will
    not be available and the coverage statistic will only be approximate.
'''.format(
    prog=sys.argv[0], 
    default_output_file=defaults.default_output_file
),
}



"""
Build
"""
def build_command():
    """Build command
    """
    if len(sys.argv) < 3:
        print(help_str['build'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('target', default=None)
    parser.add_argument('-c', '--cores', type=int, nargs='+', default=[1])
    parser.add_argument('-s', '--sets', type=int, nargs='+', default=[defaults.default_llc_sets])
    args = parser.parse_args(sys.argv[2:])

    print('Building ChampSim versions using args:')
    print('    Target:', args.target)
    print('    Cores :', args.cores)
    print('    Sets  :', args.sets)

    if args.target not in ['all'] + defaults.prefetcher_names:
        print('Invalid build target', args.target)
        exit(-1)

    # Build ChampSims with different LLC prefetchers.
    cores = set(args.cores)
    sets = set(args.sets)

    for name, fn in zip(defaults.prefetcher_names, defaults.prefetcher_fns):
        if not (args.target == 'all' or name == args.target):  # Do not build a prefetcher if it's not specified (or all)
            continue

        for c in cores:
            for s in sets:
                build.build_config(fn, c, s)


"""
Run
"""
def run_command():
    """Run command
    """
    if len(sys.argv) < 3:
        print(help_str['run'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('execution_traces', nargs='+', type=str, default=None)
    parser.add_argument('-f', '--config', type=str, default='config/prefetcher_zoo.ini')
    parser.add_argument('-t', '--targets', nargs='+', type=str, default=['no'])
    parser.add_argument('-c', '--cores', type=int, default=1)
    parser.add_argument('-s', '--sets', type=int, default=defaults.default_llc_sets)
    parser.add_argument('--results-dir', default=defaults.default_results_dir)
    parser.add_argument('--warmup-instructions', default=defaults.default_warmup_instructions)
    parser.add_argument('--num-instructions', default=defaults.default_sim_instructions)

    args = parser.parse_args(sys.argv[2:])
    
    # Assertion checks
    assert len(args.execution_traces) == args.cores, f'Provided {len(args.execution_traces)} traces for a {args.cores} core simulation.'   
    assert(not (len(args.targets) > 1 and 'no' in args.targets)), f'Cannot run "no" prefetcher in a hybrid setup: {args.targets}'
    assert(all([t in defaults.default_prefetcher_candidates for t in args.targets])), f'At least one target in {args.targets} not in {defaults.default_prefetcher_candidates}'      

    # Generate results directory
    results_dir = args.results_dir.rstrip('/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Generate names for this permutation. (trace names without extensions, joined by hyphen)
    base_traces = '-'.join(
        [''.join(os.path.basename(et).split('.')[:-2]) for et in args.execution_traces]
    )

    if args.targets == ['no']:
        llc_pref_fn = 'no'
    else:
        llc_pref_fn = 'multi'


    binary = defaults.default_binary.format(
        llc_pref_fn = llc_pref_fn, 
        llc_repl_fn = defaults.default_llc_repl_fn, 
        n_cores = args.cores
    ) + defaults.llc_sets_suffix.format(n_sets = args.sets)
        
    base_binary = os.path.basename(binary)
    assert os.path.exists(binary), f'ChampSim binary not found, (looked for {binary})'
    
    # Run ChampSim
    cmd = '{binary} {llc_pref_knobs} --warmup_instructions={warm}000000 --simulation_instructions={sim}000000 --config={config} -traces {trace} > {results}/{base_traces}-{base_binary}-{llc_pref_str}.txt 2>&1'.format(
        binary=binary,
        llc_pref_knobs=' '.join([f'--llc_prefetcher_types={t}' for t in args.targets]),
        llc_pref_str='_'.join(args.targets),
        #period=args.stat_printing_period,
        warm=args.warmup_instructions,
        sim=args.num_instructions,
        config=args.config, # .ini file
        trace=' '.join(args.execution_traces),
        results=results_dir,
        base_traces=base_traces,
        base_binary=base_binary
    )

    print('Running "' + cmd + '"')
    os.system(cmd)
    
    
    
"""
Condor Setup
"""
def condor_setup_command():
    """Condor Setup command
    """
    if len(sys.argv) < 3:
        print(help_str['condor_setup'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('prefetchers', nargs='+', type=str)
    parser.add_argument('-d', '--experiment-dir', type=str, default=defaults.default_exp_dir)
    parser.add_argument('-t', '--trace-dir', type=str, default=defaults.default_trace_dir)
    parser.add_argument('-h', '--hybrid', type=int, default=defaults.default_max_hybrid)
    parser.add_argument('-s', '--llc-sets', type=int, default=defaults.default_llc_sets)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--warmup-instructions', default=defaults.default_warmup_instructions)
    parser.add_argument('--num-instructions', default=defaults.default_sim_instructions)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    
    
    champsim_dir = defaults.default_champsim_dir if not os.environ.get('PYTHIA_HOME') else os.environ.get('PYTHIA_HOME')
    
    print('Setting up Condor Prefetcher Zoo experiment:')
    print('    ChampSim dir   :', champsim_dir)
    print('    Experiment dir :', args.experiment_dir)
    print('    Trace dir      :', args.trace_dir)
    print('    # instructions :', args.num_instructions, 'million')
    print('    # warmup       :', args.warmup_instructions, 'million')
    
    print('Cache / prefetcher setup:')
    print('    Prefetchers    :', ', '.join(args.prefetchers))
    print('    Max hybrid     :', args.hybrid)
    print('    # LLC sets     :', args.llc_sets)
    
    condor.build_sweep(
        args.trace_dir,
        args.prefetchers,
        llc_num_sets=args.llc_sets,
        exp_dir=args.experiment_dir,
        champsim_dir=champsim_dir,
        num_instructions=args.num_instructions,
        warmup_instructions=args.warmup_instructions,
        dry_run=args.dry_run,
        verbose=args.verbose
    )



"""
Eval
"""
def eval_command():
    """Eval command
    """
    pass



"""
Help
"""
def help_command():
    """Help command
    """
    # If one of the available help strings, print and exit successfully
    if len(sys.argv) > 2 and sys.argv[2] in help_str:
        print(help_str[sys.argv[2]])
        exit()
    # Otherwise, invalid subcommand, so print main help string and exit
    else:
        print(help_str['help'])
        exit(-1)



"""
Main
"""
commands = {
    'build': build_command,
    'run': run_command,
    'condor_setup': condor_setup_command,
    'eval': eval_command,
    'help': help_command,
}

def main():
    # If no subcommand specified or invalid subcommand, print main help string and exit
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(help_str['help'])
        exit(-1)

    # Run specified subcommand
    commands[sys.argv[1]]()

if __name__ == '__main__':
    main()
