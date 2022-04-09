#!/usr/bin/env python3

"""
Set up and evaluate sweeps to tune prefetchers'
degrees.

TODO : Need to implement

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

from exp_utils import defaults, condor, evaluate

# TODO: Move to config
default_max_degree = 8
default_exp_dir = '/scratch/cluster/cmolder/prefetcher_degree_sweep/exp/'
default_eval_csv = './out/prefetcher_degree_sweep.csv'

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    condor_setup     Set up Prefetcher Degree sweep on Condor
    eval             Parse and compute metrics on sweep results
    help             Display this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

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
        
    -g / --degree <max-degree>
        Will sweep all degrees from 1 to <max-degree> inclusive.
        
        Default: {default_max_degree}
    
    -h / --hybrid <max-hybrid-counts>
        Will sweep all combinations of LLC <prefetchers>, up to <max-hybrid-counts> running
        at the same time. For example, -h 2 will sweep degrees for configurations of all 2 hybrids, 
        single prefetchers, and no prefetcher.
        
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
    default_exp_dir=default_exp_dir,
    default_trace_dir=defaults.default_trace_dir,
    default_max_degree=default_max_degree,
    default_max_hybrid=defaults.default_max_hybrid,
    default_llc_sets=defaults.default_llc_sets,
    default_warmup_instructions=defaults.default_warmup_instructions,
    default_sim_instructions=defaults.default_sim_instructions
),
    
'eval': '''usage: {prog} eval <results-dir> [--output-file <output-file>] [--norm-baseline <baseline>]

Description:
    {prog} eval <results-dir>
        Runs the evaluation procedure on the ChampSim result files found in <results-dir>
        and outputs a CSV at the specified output path.

Options:
    -o / --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This defaults to
        `{default_output_file}`.
        
    --dry-run
        If passed, builds the spreadsheet but writes nothing to <output-file>.

Note:
    To get stats comparing performance to a no-prefetcher baseline, it is necessary
    to have run the base ChampSim binary on the same execution trace.

    Without the base data, relative performance data comparing MPKI and IPC will
    not be available and the coverage statistic will only be approximate.
'''.format(
    prog=sys.argv[0], 
    default_output_file=default_eval_csv
),
}


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
    parser.add_argument('-d', '--experiment-dir', type=str, default=default_exp_dir)
    parser.add_argument('-t', '--trace-dir', type=str, default=defaults.default_trace_dir)
    parser.add_argument('-g', '--degree', type=int, default=default_max_degree)
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
    print('    Max degree     :', args.degree)
    print('    Max hybrid     :', args.hybrid)
    print('    # LLC sets     :', args.llc_sets)
    
    condor.build_degree_sweep(
        args.trace_dir,
        args.prefetchers,
        args.degree,
        max_hybrid=args.hybrid,
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
    
    Return the best degree prefetcher for each prefetcher and phase.
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('-o', '--output-file', type=str, default=default_eval_csv)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    
    evaluate.generate_best_degree_csv(
        args.results_dir,
        args.output_file,
        dry_run=args.dry_run
    )



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
