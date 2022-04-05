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

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    condor_setup     Set up Prefetcher Zoo sweep on Condor
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
    default_output_file=defaults.default_output_file
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

    pass



"""
Eval
"""
def eval_command():
    """Eval command
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('-o', '--output-file', type=str, default=defaults.default_output_file)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    
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
