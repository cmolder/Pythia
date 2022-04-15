#!/usr/bin/env python3

"""
Set up and evaluate sweeps for the Prefetcher Zoo
experiments.

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

from exp_utils import condor, config, evaluate, pc_trace

# Defaults (TODO: Move to yml or launch args)
default_eval_csv = './out/prefetcher_zoo.csv'
default_pc_trace_metric = 'num_useful'

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    condor    Set up Prefetcher Zoo sweep on Condor
    eval      Parse and compute metrics on sweep results
    pc_trace  Parse a per-PC statistics file and generate traces of the
              best prefetchers for each PC on each SimPoint
    help      Display this help message. Command-specific help messages
              can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'condor': '''usage: {prog} condor <config-file> [-v / --verbose] [-d / --dry-run]

Description:
    {prog} condor <config-file>
        Sets up a Prefetching Zoo sweep for use on Condor. <config-file> is a path to a 
        .yml file with the config (example: experiments/exp_utils/zoo.yml)
        
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
        (i.e. champsim_results/) and outputs a CSV at the specified output path.

Options:
    -o / --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This defaults to
        `{default_eval_csv}`.
        
    --best-degree-csv <best-degree-output-file>
        If provided, will create <prefetcher>-best variants that use the tuned
        version of each prefetcher, using the results from <best-degree-output-file>
        in prefetcher_degree_sweep. 
        
        Must copy the relevant degree prefetcher result files from the degree sweep 
        to <results-dir>.
        
    --pc
        If provided, will compute per-PC prefetch stats on the LLC, using results
        in <results-dir>/pc_pref_stats/
        
    --dry-run
        If passed, builds the spreadsheet but writes nothing to <output-file>.

Note:
    To get stats comparing performance to a no-prefetcher baseline, it is necessary
    to have run the base ChampSim binary on the same execution trace.

    Without the base data, relative performance data comparing MPKI and IPC will
    not be available and the coverage statistic will only be approximate.
'''.format(
    prog=sys.argv[0], 
    default_eval_csv=default_eval_csv
),
'pc_trace': '''usage: {prog} pc_trace <pc-stats-file> <output-dir> [-m / --metric <metric>]

Description:
    {prog} pc_trace <pc-stats-file> <output-dir>
        Parses a PC stats file, and for each PC in each trace, determines the best
        prefetcher under <metric>. These traces are saved to the output dir, for use
        in the multi_pc_trace prefetcher.
        
Options:
    -m / --metric <metric>
        Specifies what metric to evaluate prefetchers by. Currently, the options are:
        {metric_options}
'''.format(
    prog=sys.argv[0],
    metric_options=pc_trace.metrics
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
    
    print('Setting up Condor Prefetcher Zoo experiment:')
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
    print()
    
    condor.build_sweep(cfg, dry_run=args.dry_run, verbose=args.verbose)



"""
Eval
"""
def eval_command():
    """Eval command
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('-o', '--output-file', type=str, default=default_eval_csv)
    parser.add_argument('--best-degree-csv', type=str)
    parser.add_argument('--pc', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    
    print('Generating cumulative run statistics...')
    evaluate.generate_csv(
        args.results_dir,
        args.output_file,
        best_degree_csv_file=args.best_degree_csv,
        dry_run=args.dry_run
    )
    
    if args.pc:
        print('Generating per-PC run statistics...')
        evaluate.generate_pc_csv(
            args.results_dir,
            args.output_file.replace('.csv', '_pc_llc.csv'),
            level='llc',
            best_degree_csv_file=args.best_degree_csv,
            dry_run=args.dry_run
        )
        
        
"""
PC Trace
"""
def pc_trace_command():
    """PC trace command
    """ 
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('pc_stats_file', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-m', '--metric', type=str, default=default_pc_trace_metric)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    
    pc_trace.build_pc_traces(
        args.pc_stats_file,
        args.output_dir,
        args.metric,
        level='llc',
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
    'pc_trace': pc_trace_command,
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
