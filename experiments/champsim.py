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

'build': '''usage: {prog} build [--l1d-pref <l1d-prefetcher>] [--l2c-pref <l2c-prefetcher>] [--llc-pref <llc-prefetcher>]
                                [-c / --cores <core-count-list>] [-s / --llc-sets <llc-set-count-list>]

Description:
    {prog} build 
        Builds ChampSim binaries.

        Note: The replacement policy is SHiP for the LLC.

Options:
    --branch-pred <branch-predictor>
        Choose the branch predictor, where <branch-pred> is one of:
        
        perceptron          Branch perceptron (Default)
        hashed_perceptron   Hashed Branch Perceptron
        gshare              Gshare
        bimodal             Bimodal

    --l1d-pref <l1d-prefetcher>
        Choose the L1D prefetcher, where <l1d-prefetcher> is one of:
        
        no             No prefetcher (Default)
        multi          Runtime-configurable prefetcher (see multi.l1d_pref for options)

    --l2c-pref <l2c-prefetcher>
        Choose the L2 prefetcher, where <llc-prefetcher> is one of:
        
        no             No prefetcher (Default)
        multi          Runtime-configurable prefetcher (see multi.l2c_pref for options)

    --llc-pref <llc-prefetcher>
        Choose the LLC prefetcher, where <llc-prefetcher> is one of:
        
        no             No prefetcher (Default)
        multi          Runtime-configurable prefetcher (see multi.llc_pref for options)
        multi_pc_trace Prefetch using the chosen prefetcher for each PC, as defined in 
                       a PC trace.
                       
   --llc-repl <llc-replacement>
        Choose the LLC replacement policy, where <llc-replacement> is one of:
        
        ship           Signature-based Hit Predictor (SHiP) (Default)
        srrip          Static RRIP
        drrip          Dynamic RRIP
        lru            Least recently used

    -c / --cores <core-count-list>
        Specifies a list of cores to build ChampSim variants. A single core
        version will always be built, but additional versions (e.g. 2-core / 4-core)
        can be listed here (e.g. using -c 2 4). The ChampSim script is tested up
        to 8 cores.
        Default: One core only

    -s / --llc-sets <llc-set-count-list>
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
    Simulator/cache options:
        -k / --knobs <knobs_file>
            The default knobs configuration to use. This
            defaults to `{default_knobs_file}`.

        -c / --cores <num-cores>
            The number of cores that ChampSim will be simulating. Must provide a <cores>
            length list of execution traces to the script. By default, one core is used.

        -s / --llc-sets <num-llc-sets>
            The number of LLC cache sets that ChampSim will be simulating. By default,
            {default_llc_sets} sets are used (if the binary is available).

        --results-dir <results-dir>
            Specifies what directory to save the ChampSim results file in. This
            defaults to `{default_results_dir}`.

        --warmup-instructions <warmup-instructions>
            Number of instructions to warmup the simulation for. Defaults to
            {default_warmup_instructions}M instructions

        --num-instructions <num-instructions>
            Number of instructions to run the simulation for. Defaults to
            {default_sim_instructions}M instructions
            
        -p / --track-pc
            Track per-PC prefetch statistics, and save them to a file inside <results-dir>/pc-pref-stats.
            
        -a / --track-addr
            Track per-address prefetch statistics, and save them to a file inside <results-dir>/addr-pref-stats.

    Prefetcher options:
        -t / --llc-pref <list-of-llc-prefetchers>
            List of LLC prefetchers to run. If two or more are proivded, runs them
            in a hybrid setting. By default, it will run no prefetcher.
            
        --l2c-pref <list-of-l2c-prefetchers>
        
        --l1d-pref <list-of-l1d-prefetchers>
        
        --llc-pref-degrees <list-of-llc-prefetcher-degrees>
            List of degrees to run each LLC prefetcher. If the prefetcher does not
            support variable degrees, the value is ignored. Pass them in the same
            order as `--llc-pref`. 

            Defaults to the knobs in the config file
            passed to `--config`, or the values in src/knobs.cc if the relevant knob 
            isn't provided in the config file.
            
        --l2c-pref-degrees <list-of-l2c-prefetcher-degrees>
        
        --pc-trace-llc <pc-trace-file>
            File to a PC trace. Must be passed if the target is 'pc_trace'.
            
    Replacement options:
        --llc-repl <llc-replacement>
            LLC replacement policy to use. By default, SHiP ("ship") is used.
            
    Branch prediction options:
        --branch-pred <branch-predictor>
            Branch predictor to use. By default, Branch Perceptron ("perceptron") is used.
        

    

'''.format(
    prog=sys.argv[0], 
    default_knobs_file=defaults.default_knobs_file,
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
    if len(sys.argv) < 2:
        print(help_str['build'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('--branch-pred', default='perceptron')
    parser.add_argument('--l1d-pref', default='no')
    parser.add_argument('--l2c-pref', default='no')
    parser.add_argument('--llc-pref', default='no')
    parser.add_argument('--llc-repl', default='ship')
    parser.add_argument('-c', '--cores', type=int, nargs='+', default=[1])
    parser.add_argument('-s', '--llc-sets', type=int, nargs='+', default=[defaults.default_llc_sets])
    args = parser.parse_args(sys.argv[2:])

    print('Building ChampSim versions using args:')
    print('    Branch predictor:', args.branch_pred)
    print('    L1D prefetcher  :', args.l1d_pref)
    print('    L2C prefetcher  :', args.l2c_pref)
    print('    LLC prefetcher  :', args.llc_pref)
    print('    LLC replacement :', args.llc_repl)
    print('    # Cores         :', args.cores)
    print('    # LLC Sets      :', args.llc_sets)
    
    if args.l1d_pref not in defaults.l1d_pref_fns:
        print('Invalid l1d prefetcher', args.l1d_pref)
        exit(-1)
    if args.l2c_pref not in defaults.l2c_pref_fns:
        print('Invalid l2c prefetcher', args.l2c_pref)
        exit(-1)
    if args.llc_pref not in defaults.l2c_pref_fns:
        print('Invalid llc prefetcher', args.llc_pref)
        exit(-1)

    # Build ChampSims with different core / set counts.
    cores = set(args.cores)
    llc_sets = set(args.llc_sets)

    for c in cores:
        for s in llc_sets:
            build.build_config(
                c, 
                branch_pred=args.branch_pred,
                l1d_pref=args.l1d_pref,
                l2c_pref=args.l2c_pref,
                llc_pref=args.llc_pref,
                llc_repl=args.llc_repl,
                llc_num_sets=s
            )


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
    
    # Sim / cache options
    parser.add_argument('execution_traces', nargs='+', type=str, default=None)
    parser.add_argument('-k', '--knobs', type=str, default=defaults.default_knobs_file)
    parser.add_argument('-c', '--cores', type=int, default=1)
    parser.add_argument('-s', '--llc-sets', type=int, default=defaults.default_llc_sets)
    parser.add_argument('--results-dir', default=defaults.default_results_dir)
    parser.add_argument('--warmup-instructions', default=defaults.default_warmup_instructions)
    parser.add_argument('--num-instructions', default=defaults.default_sim_instructions)
    parser.add_argument('-p', '--track-pc', action='store_true')
    parser.add_argument('-a', '--track-addr', action='store_true')
    
    # Prefetcher options
    parser.add_argument('-t', '--llc-pref', nargs='+', type=str, default=['no'])
    parser.add_argument('--l2c-pref', nargs='+', type=str, default=['no'])
    parser.add_argument('--l1d-pref', nargs='+', type=str, default=['no'])
    parser.add_argument('--llc-pref-degrees', nargs='+', type=int, default=[])
    parser.add_argument('--l2c-pref-degrees', nargs='+', type=int, default=[])
    parser.add_argument('--pc-trace-llc', type=str, default=None)
    # No support for l1d degree
    
    # Replacement options
    parser.add_argument('--llc-repl', type=str, default='ship')
    
    # Branch prediction options
    parser.add_argument('--branch-pred', type=str, default='perceptron')

    args = parser.parse_args(sys.argv[2:])
    
    # Assertion checks
    assert len(args.execution_traces) == args.cores, f'Provided {len(args.execution_traces)} traces for a {args.cores} core simulation.'   
    assert(not (len(args.llc_pref) > 1 and 'no' in args.llc_pref)), f'Cannot run "no" prefetcher in an LLC hybrid setup: {args.llc_pref}'
    assert(not (len(args.l2c_pref) > 1 and 'no' in args.l2c_pref)), f'Cannot run "no" prefetcher in an L2C hybrid setup: {args.l2c_pref}'
    assert(not (len(args.l1d_pref) > 1 and 'no' in args.l1d_pref)), f'Cannot run "no" prefetcher in an L1D hybrid setup: {args.l1d_pref}'
    if args.llc_pref == ['pc_trace']:
        assert args.pc_trace_llc is not None, 'Must pass a pc_trace file to <pc-trace-llc> if running pc_trace prefetcher.'
    

    # Generate results directory
    results_dir = args.results_dir.rstrip('/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    # Choose llc prefetcher binary
    l1d_pref_fn = run.get_llc_pref_fn(args.l1d_pref)
    l2c_pref_fn = run.get_l2c_pref_fn(args.l2c_pref)
    llc_pref_fn = run.get_l1d_pref_fn(args.llc_pref)

    
    # Generate paths
    binary = run.get_binary(
        branch_pred=args.branch_pred,
        l1d_pref=l1d_pref_fn,
        l2c_pref=l2c_pref_fn,
        llc_pref=llc_pref_fn,
        llc_repl=args.llc_repl,
        n_cores=args.cores, 
        llc_n_sets=args.llc_sets, 
    )
    results_file = run.get_results_file(
        binary, args.execution_traces, # Infer branch predictor, llc replacement from binary path.
        l1d_prefs=args.l1d_pref,
        l2c_prefs=args.l2c_pref,
        llc_prefs=args.llc_pref,
        l2c_pref_degrees=args.l2c_pref_degrees,
        llc_pref_degrees=args.llc_pref_degrees
    )
    assert os.path.exists(binary), f'ChampSim binary not found, (looked for {binary})'
    
    # Run ChampSim
    # NOTE: Put config knob first, so any other added knobs override it.
    cmd = '{binary} --config={config} --warmup_instructions={warm}000000 --simulation_instructions={sim}000000 {cloudsuite_knobs} {l1d_pref_knobs} {l2c_pref_knobs} {llc_pref_knobs} {out_trace_knobs} {pc_trace_knobs} -traces {trace} > {results}/{results_file} 2>&1'.format(
        binary=binary,
        cloudsuite_knobs=run.get_cloudsuite_knobs(args.execution_traces),
        l1d_pref_knobs=run.get_prefetcher_knobs(args.l1d_pref, level='l1d'),
        l2c_pref_knobs=run.get_prefetcher_knobs(args.l2c_pref, pref_degrees=args.l2c_pref_degrees, level='l2c'),
        llc_pref_knobs=run.get_prefetcher_knobs(args.llc_pref, pref_degrees=args.llc_pref_degrees, level='llc'),
        out_trace_knobs=run.get_output_trace_knobs(results_dir, results_file, track_pc=args.track_pc, track_addr=args.track_addr),
        pc_trace_knobs=f' --pc_trace_llc={args.pc_trace_llc}' if args.pc_trace_llc else '',
        #period=args.stat_printing_period,
        warm=args.warmup_instructions,
        sim=args.num_instructions,
        config=args.knobs, # .ini file
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
