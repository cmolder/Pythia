#!/usr/bin/env python3

"""
Build and run singlecore versions of ChampSim.

Authors: Quang Duong and Carson Molder
"""

# Try not to import anything outside Python default libraries.
import argparse
import os
import sys
import shutil
import itertools
from collections import defaultdict

from exp_utils import defaults, build, run

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
    help             Display this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'build': '''usage: {prog} build <target> [-c / --cores <core-count-list>] [-s / --sets <llc-set-count-list>]

Description:
    {prog} build <prefetcher>
        Builds ChampSim binaries with an LLC <prefetcher>, where <prefetcher> is one of:

        no             No prefetcher
        multi          Runtime-configurable prefetcher (see multi.llc_pref for options)
        multi_pc_trace Prefetch using the chosen prefetcher for each PC, as defined in 
                       a PC trace.
        all            All of the above.
        
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

'run': '''usage: {prog} run <execution-traces> [-c / --cores <num-cores>] [-s / --sets <num-llc-sets>]
                            [-t / --llc-pref <list-of-llc-prefetchers>]
                            [--llc-pref-degrees <list-of-llc-prefetcher-degrees>]
                            [--results-dir <results-dir>] 
                            [--warmup-instructions <warmup-instructions>] 
                            [--num-instructions <num-instructions>] 

Description:
    {prog} run <execution-traces>
        Runs the base ChampSim binary on the specified execution trace(s). If using
        a multi-core setup, must provide <cores> traces.

Options:
    -f / --config <config_file>
        The knobs configuration to use. This
        defaults to `{default_config_file}`.
    
    -c / --cores <num-cores>
        The number of cores that ChampSim will be simulating. Must provide a <cores>
        length list of execution traces to the script. By default, one core is used.

    -s / --sets <num-llc-sets>
        The number of LLC cache sets that ChampSim will be simulating. By default,
        {default_llc_sets} sets are used (if the binary is available).

    -t / --llc-pref <list-of-llc-prefetchers>
        List of LLC prefetchers to run. If two or more are proivded, runs them
        in a hybrid setting. By default, it will run no prefetcher.
        
    --pc-trace-llc <pc-trace-file>
        File to a PC trace. Must be passed if the target is 'pc_trace'.
        
    --llc-pref-degrees <list-of-llc-prefetcher-degrees>
        List of degrees to run each LLC prefetcher. If the prefetcher does not
        support variable degrees, the value is ignored. Pass them in the same
        order as `--llc-prefetchers`. 
        
        Defaults to the knobs in the config file
        passed to `--config`, or the values in knobs.cc if the relevant knob 
        isn't provided in the config file.
    
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
    default_config_file=defaults.default_config_file,
    default_results_dir=defaults.default_results_dir,
    default_warmup_instructions=defaults.default_warmup_instructions,
    default_sim_instructions=defaults.default_sim_instructions,
    default_llc_sets=defaults.default_llc_sets,
)}



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
    parser.add_argument('-f', '--config', type=str, default=defaults.default_config_file)
    parser.add_argument('-t', '--llc-pref', nargs='+', type=str, default=['no'])
    parser.add_argument('-c', '--cores', type=int, default=1)
    parser.add_argument('-s', '--sets', type=int, default=defaults.default_llc_sets)
    parser.add_argument('--pc-trace-llc', type=str, default=None)
    parser.add_argument('--llc-pref-degrees', nargs='+', type=int, default=[])
    parser.add_argument('--results-dir', default=defaults.default_results_dir)
    parser.add_argument('--warmup-instructions', default=defaults.default_warmup_instructions)
    parser.add_argument('--num-instructions', default=defaults.default_sim_instructions)

    args = parser.parse_args(sys.argv[2:])
    
    # Assertion checks
    assert len(args.execution_traces) == args.cores, f'Provided {len(args.execution_traces)} traces for a {args.cores} core simulation.'   
    assert(not (len(args.llc_pref) > 1 and 'no' in args.llc_pref)), f'Cannot run "no" prefetcher in a hybrid setup: {args.llc_pref}'
    if args.llc_pref == ['pc_trace']:
        assert args.pc_trace_llc is not None, 'Must pass a pc_trace file to <pc-trace-llc> if running pc_trace prefetcher.'
    

    # Generate results directory
    results_dir = args.results_dir.rstrip('/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    # Choose llc prefetcher binary
    if args.llc_pref == ['no']:
        llc_pref_fn = 'no'
    elif args.llc_pref == ['pc_trace']:
        llc_pref_fn = 'multi_pc_trace'
    else:
        llc_pref_fn = 'multi'
    
    # Generate paths
    binary = run.get_binary(
        llc_pref_fn=llc_pref_fn, 
        llc_repl_fn=defaults.default_llc_repl_fn, 
        n_cores=args.cores, 
        llc_n_sets=args.sets, 
    )
    results_file = run.get_results_file(
        binary, args.execution_traces, 
        llc_prefs=args.llc_pref,
        llc_pref_degrees=args.llc_pref_degrees
    )
    assert os.path.exists(binary), f'ChampSim binary not found, (looked for {binary})'
    
    # Run ChampSim
    # NOTE: Put config knob first, so any other added knobs override it.
    cmd = '{binary} --config={config} --warmup_instructions={warm}000000 --simulation_instructions={sim}000000 {cloudsuite_knobs} {llc_pref_knobs} {pc_pref_knobs} {pc_trace_knobs} -traces {trace} > {results}/{results_file} 2>&1'.format(
        binary=binary,
        cloudsuite_knobs=run.get_cloudsuite_knobs(args.execution_traces),
        llc_pref_knobs=run.get_prefetcher_knobs(args.llc_pref, pref_degrees=args.llc_pref_degrees),
        pc_pref_knobs=run.get_pc_prefetcher_knobs(results_dir, results_file),
        pc_trace_knobs=f' --pc_trace_llc={args.pc_trace_llc}' if args.pc_trace_llc else '',
        #period=args.stat_printing_period,
        warm=args.warmup_instructions,
        sim=args.num_instructions,
        config=args.config, # .ini file
        trace=' '.join(args.execution_traces),
        results=results_dir,
        results_file=results_file
    )

    print('Running "' + cmd + '"')
    os.system(cmd)
    

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
