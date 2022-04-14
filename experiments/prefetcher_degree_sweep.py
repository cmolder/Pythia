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

from exp_utils import config, condor, evaluate

# Defaults (TODO: Move to yml or launch args)
default_eval_csv = './out/prefetcher_degree_sweep.csv'

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    condor Set up Prefetcher Degree sweep on Condor
    eval   Parse and compute metrics on sweep results
    help   Display this help message. Command-specific help messages
           can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'condor': '''usage: {prog} condor <config-file> [-v / --verbose] [-d / --dry-run]

Description:
    {prog} condor <config-file>
        Sets up a Prefetching Degree sweep for use on Condor. <config-file> is a path to a 
        .yml file with the config (example: experiments/exp_utils/degree.yml)
        
Options:
    -v / --verbose
        If passed, prints extra details about the experiment setup.
        
    -d / --dry-run
        If passed, builds the experiment but writes nothing to <experiment-dir>.
'''.format(
    prog=sys.argv[0], 
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
Condor
"""
def condor_command():
    """Condor command
    """
    if len(sys.argv) < 3:
        print(help_str['condor'])
        exit(-1)
        
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('config_file', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    cfg = config.read_config(args.config_file)
    
    print('Setting up Condor Degree Sweep experiment:')
    print('    ChampSim:')
    print('        # sim inst    :', cfg.champsim.sim_instructions, 'million')
    print('        # warmup inst :', cfg.champsim.warmup_instructions, 'million')
    print('    Directories:')
    print('        ChampSim   :', cfg.paths.champsim_dir)
    print('        Experiment :', cfg.paths.exp_dir)
    print('        Traces     :', cfg.paths.trace_dir)
    print('    LLC:')
    print('        Sets             :', cfg.llc.sets)
    print('        Pref. candidates :', ', '.join(cfg.llc.pref_candidates))
    print('        Max hybrid       :', cfg.llc.max_hybrid)
    print('        Max degree       :', cfg.llc.max_degree)
    print()
    
    condor.build_degree_sweep(cfg, dry_run=args.dry_run, verbose=args.verbose)



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
    'condor': condor_command,
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
