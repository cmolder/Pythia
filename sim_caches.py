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


default_llc_repl_fn = 'ship'
default_results_dir = './out/results'
default_output_file = './out/stats.csv'
default_instrs = 50
default_printing_period_instrs = 10

default_llc_ways = 16
default_llc_sets = 2048

default_binary = 'bin/perceptron-no-no-{llc_pref_fn}-{llc_repl_fn}-{n_cores}core'
llc_sets_suffix = '-{n_sets}llc_sets'
llc_ways_suffx = '-{n_ways}llc_ways'

#prefetcher_names = ['no', 'bo', 'bingo', 'misb', 'pythia', 'spp', 'sms', 'triage']
#prefetcher_fns = ['no', 'bop', 'bingo', 'misb', 'scooby', 'spp_dev2', 'sms', 'triage'] # TODO - Double check in prefetcher/
prefetcher_names = ['no', 'multi']
prefetcher_fns = ['no', 'multi']


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
    eval             Parse and compute metrics on simulation results
    help             Display this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'build': '''usage: {prog} build <target> [-c / --cores <core-count-list>] [-s / --sets <llc-set-count-list>]
                                [--hawkeye-splits <hawkeye_split_list>]

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
'''.format(prog=sys.argv[0], llc_ways=default_llc_ways, default_llc_sets=default_llc_sets),

'run': '''usage: {prog} run <execution-traces> [-c / --cores <num-cores>] [-s / --sets <num-llc-sets>]
                            [-t / --targets <list-of-targets>] [--hawkeye-split <hawkeye-split>]
                            [--results-dir <results-dir>] [--num-instructions <num-instructions>] 
                            [--stat-printing-period <num-instructions>]

TODO - WORK IN PROGRESS FOR PREFETCHER ZOO! NOT READY FOR USE!

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
        
    --hawkeye-split <hawkeye-split>
        Split of ways between the <n_cores> OPTgens. If not provided, hawkeye_split
        will be skipped.

    --results-dir <results-dir>
        Specifies what directory to save the ChampSim results file in. This
        defaults to `{default_results_dir}`.

    --num-instructions <num-instructions>
        Number of instructions to run the simulation for. Defaults to
        {default_instrs}M instructions

    --stat-printing-period <num-instructions>
        Number of instructions to simulate between printing out statistics.
        Defaults to {default_printing_period_instrs}M instructions.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir,
    prefetcher_names=prefetcher_names,
    default_instrs=default_instrs,
    default_llc_sets=default_llc_sets,
    default_printing_period_instrs=default_printing_period_instrs),

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
'''.format(prog=sys.argv[0], default_output_file=default_output_file),
}



"""
Build
"""
def change_llc_sets(cacheh_path, num_cpus, num_sets):
    """Replace the number of sets in the ChampSim LLC definition."""
    print(f'Changing LLC sets in inc/cache.h to NUM_CPUS*{num_sets} (effectively {num_cpus} * {num_sets}, {num_cpus*num_sets*16 / 1024} KB)...')

    replacement = ''
    with open(cacheh_path, 'rt') as f:
        for line in f:
            if 'LLC_SET' in line:
                line = f'#define LLC_SET NUM_CPUS*{num_sets}\n'
            replacement += line

    with open(cacheh_path, 'wt') as f:
        print(replacement, file=f)


def build_binary(llc_pref_fn, num_cpus):
    os.system(f'./build_champsim.sh no no {llc_pref_fn} {num_cpus}')


def backup_file(path):
    if os.path.exists(path):
        print(f'Backing up {path}...')
        shutil.copyfile(path, path + '.bak')


def restore_file(path):
    if os.path.exists(path + '.bak'):
        print(f'Restoring {path} from backup...')
        shutil.copyfile(path + '.bak', path)
        os.remove(path + '.bak')


def move_file(old_path, new_path):
    if os.path.exists(old_path):
        print(f'Moving {old_path} to {new_path}...')
        shutil.move(old_path, new_path)


def build_config(llc_pref_fn, num_cpus, num_sets):
    print(f'=== Building "{llc_pref_fn}" ChampSim binary, {num_cpus} core{"s" if num_cpus > 1 else ""}, {num_sets} LLC sets ===')

    # Backup files
    backup_file('./inc/cache.h') # Backup original cache.h file
    old_binary = default_binary.format(llc_pref_fn=llc_pref_fn, llc_repl_fn=default_llc_repl_fn, n_cores=num_cpus)
    new_binary = old_binary + f'-{num_sets}llc_sets'
    backup_file(old_binary)      # Backup original binary (if one clashes with ChampSim's output)

    # Modify files and build
    change_llc_sets('./inc/cache.h', num_cpus, num_sets) # Change cache.h file to accomodate desired number of sets
    build_binary(llc_pref_fn, num_cpus)               # Build ChampSim with modified cache.h
    move_file(old_binary, new_binary)                    # Rename new binary to reflect changes.

    # Restore backups
    restore_file('./inc/cache.h')                        # Restore original cache.h file.
    restore_file(old_binary)                             # Restore original binary (if one exists)


def build_command():
    """Build command
    """
    if len(sys.argv) < 3:
        print(help_str['build'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('target', default=None)
    parser.add_argument('-c', '--cores', type=int, nargs='+', default=[1])
    parser.add_argument('-s', '--sets', type=int, nargs='+', default=[default_llc_sets])
    args = parser.parse_args(sys.argv[2:])

    print('Building ChampSim versions using args:')
    print('    Target:', args.target)
    print('    Cores :', args.cores)
    print('    Sets  :', args.sets)

    if args.target not in ['all'] + prefetcher_names:
        print('Invalid build target', args.target)
        exit(-1)

    # Build ChampSims with different LLC prefetchers.
    cores = set(args.cores)
    sets = set(args.sets)

    for name, fn in zip(prefetcher_names, prefetcher_fns):
        if not (args.target == 'all' or name == args.target):  # Do not build a prefetcher if it's not specified (or all)
            continue

        for c in cores:
            for s in sets:
                build_config(fn, c, s)


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
    parser.add_argument('-t', '--targets', nargs='+', type=str, default=replacement_names)
    parser.add_argument('-c', '--cores', type=int, default=1)
    parser.add_argument('-s', '--sets', type=int, default=default_llc_sets)
    parser.add_argument('--hawkeye-split', nargs='+', type=int, default=None)
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--num-instructions', default=500) #None) #default_spec_instrs if execution_trace[0].isdigit() else default_gap_instrs)
    parser.add_argument('--stat-printing-period', default=default_printing_period_instrs)

    args = parser.parse_args(sys.argv[2:])
    assert len(args.execution_traces) == args.cores, f'Provided {len(args.execution_traces)} traces for a {args.cores} core simulation.'
    execution_traces = args.execution_traces

    # Generate results directory
    results_dir = args.results_dir #os.path.join(args.results_dir, f'{args.sets}llc_sets')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Generate names for this permutation. (trace names without extensions, joined by hyphen)
    base_traces = '-'.join(
        [''.join(os.path.basename(et).split('.')[:-2]) for et in execution_traces]
    )

    for name, fn in zip(prefetcher_names, prefetcher_fns):
        
        # Retool fn to fit defined Hawkeye split, if necessary.
        if 'hawkeye_split' in name: 
            if args.hawkeye_split is None:
                print(f'Skipping hawkeye_split, no split provided to argment <hawkeye-split>.')
                continue
  
            assert args.cores > 1, 'Can only run hawkeye_split on multi-core configurations.'
            assert len(args.hawkeye_split) == args.cores, 'Must provide a split set for each core.'
        
            fn = fn + '_' + '_'.join(str(i) for i in args.hawkeye_split)
            

        binary = default_binary_sets.format(llc_pref_fn = fn, llc_repl_fn = default_llc_repl_fn, n_cores = args.cores) + llc_sets_suffix.format(n_sets = args.sets)
        base_binary = os.path.basename(binary)

        # Check if we should actually run this baseline
        if name not in args.targets:
            print(f'Skipping {name} ({binary})')
            continue

        if not os.path.exists(binary):
            print(f'{name} ChampSim binary not found, (looked for {binary})')
            exit(-1)

        
        cmd = '{binary} --warmup_instructions {warm}000000 --simulation_instructions {sim}000000 --config={config} -traces {trace} > {results}/{base_traces}-{base_binary}.txt 2>&1'.format(
            binary=binary,
            #period=args.stat_printing_period,
            warm=args.warmup_instructions,
            sim=args.num_instructions,
            config=args.config, # .ini file
            trace=' '.join(execution_traces),
            results=results_dir,
            base_traces=base_traces,
            base_binary=base_binary
        )

        print('Running "' + cmd + '"')
        os.system(cmd)


"""
Eval
"""
def get_traces_per_cpu(path):
    """Read a single ChampSim output file and get the traces on each CPU.
    """
    traces = {}
    with open(path, 'r') as f:
        for line in f:
            if 'CPU' in line and 'runs' in line:
                core = int(line.split()[1])
                #traces[core] = os.path.basename(line.split()[-1]).split('.')[0] # Trace name - TODO check this works for all traces.
                traces[core] = os.path.basename(line.split()[-1])  # File name
                #traces[core] = line.split()[-1] # Full path to file
    return traces


def read_file(path, cache_level='LLC'):
    """Read a single ChampSim output file and parse the results.
    """
    #expected_keys = ('trace', 'ipc', 'total_miss', 'useful', 'useless', 'uac_correct', 'iss_prefetches', 'load_miss', 'rfo_miss', 'kilo_inst')
    expected_keys = ('trace', 'is_homogeneous', 'llc_sets', 'ipc', 'kilo_inst', 'load_miss', 'rfo_miss', 'total_miss')


    #data = defaultdict(lambda: defaultdict(int)) # Indexed by core -> feature
    data = defaultdict(dict)

    # Build trace list
    data['trace'] = get_traces_per_cpu(path)
    data['is_homogeneous'] = len(set(data['trace'].values())) == 1

    # Build other features
    with open(path, 'r') as f:
        for line in f:
            if 'LLC sets' in line:
                llc_sets = int(line.split()[2])
                data['llc_sets'] = llc_sets
            # Finished CPU indicators
            if 'Finished CPU' in line:
                core = int(line.split()[2])
                data['ipc'][core] = float(line.split()[9])
                data['kilo_inst'][core] = int(line.split()[4]) / 1000

            # Region of interest statistics
            if 'CPU' in line and line.split()[0] == 'CPU':
                core = int(line.split()[1])
            if cache_level not in line:
                continue
            line = line.strip()
            if 'LOAD' in line:
                data['load_miss'][core] = int(line.split()[-1])
            elif 'RFO' in line:
                data['rfo_miss'][core] = int(line.split()[-1])
            elif 'TOTAL' in line:
                data['total_miss'][core] = int(line.split()[-1])
            # elif 'USEFUL' in line:
            #     data['useful'][core] = int(line.split()[-6])
            #     data['useless'][core] = int(line.split()[-4])
            #     data['uac_correct'][core] = int(line.split()[-1])
            #     data['iss_prefetches'][core] = int(line.split()[-8])

    if not all(key in data for key in expected_keys):
        return None

    return data



def compute_stats(trace_path, baseline_name=''):
    """Compute additional statistics, after reading the raw
    data from the trace. Return it as a CSV row.
    """
    data = read_file(trace_path)
    if not data:
        return pd.DataFrame({})

    n_cores = max(data['ipc'].keys()) + 1
    out = defaultdict(list)

    for core in sorted(data['ipc'].keys()):
        trace, llc_sets, llc_sets_per_core, ipc, load_miss, rfo_miss, kilo_inst = (
            data['trace'][core], data['llc_sets'], int(data['llc_sets'] / n_cores), data['ipc'][core], data['load_miss'][core],
            data['rfo_miss'][core], data['kilo_inst'][core]
        )
        is_homogeneous = data['is_homogeneous']
        mpki = (load_miss + rfo_miss) / kilo_inst

        out['Trace'].append(trace)
        out['Baseline'].append(baseline_name)
        out['CPU'].append(core)
        out['HomogeneousMix'].append(is_homogeneous)
        out['NumCPUs'].append(n_cores)
        out['LLCSets'].append(llc_sets)
        out['LLCSetsPerCPU'].append(llc_sets_per_core)
        out['MPKI'].append(mpki)
        out['NumInstMillions'].append(kilo_inst * 1000)
        out['IPC'].append(ipc)
        out['CPI'].append(1 / ipc)
        out['LoadMisses'].append(load_miss)
        out['RFOMisses'].append(rfo_miss)
        out['RunName'].append(os.path.basename(trace_path))
        
        if 'hawkeye_split' in baseline_name:
            out['HawkeyeSplitAllocation'].append(baseline_name.split('_')[-(n_cores - core)])
        else:
            out['HawkeyeSplitAllocation'].append(np.nan)

    return pd.DataFrame(out)




def default_norm_baseline(baseline, norm='Homo'):
    #print('[DEBUG] default_norm_baseline:', baseline)
    if baseline.startswith('hawkeye'):
        return 'hawkeye_simple'
    if baseline == 'ucp':
        return 'lru' #if norm == 'Single' else 'ucp'
    
    return baseline # lru



def add_homo_norm_data(df, target, norm_baseline=None):
    """Compute the homogeneous-mix run normalized
    statistics for each core/run, after other stats
    have been calcuated.
    """
    df = df.reset_index(drop=True)
    
    # Get Homo Norm of Targets
    # NOTE: (Baseline of a hawkeye_split is a hawkeye_simple)
    for i, core in df.iterrows():
        
        if norm_baseline == None:
            core_norm_baseline = default_norm_baseline(core.Baseline, norm='Homo')
        else:
            core_norm_baseline = norm_baseline

        homo_run = df[
            (df.Trace == core.Trace) &
            (df.Baseline == core_norm_baseline) &
            (df.NumCPUs == core.NumCPUs) &
            (df.LLCSetsPerCPU == core.LLCSetsPerCPU) &
            (df.HomogeneousMix == True)
        ]
        
        # For calculating FairUtility constraint on this particular core,
        # without introducing bias w.r.t a different normalization baseline.
        homo_run_default_norm = df[
            (df.Trace == core.Trace) &
            (df.Baseline == default_norm_baseline(core.Baseline, norm='Homo')) &
            (df.NumCPUs == core.NumCPUs) &
            (df.LLCSetsPerCPU == core.LLCSetsPerCPU) &
            (df.HomogeneousMix == True)
        ]
        
        df.loc[i, 'HomoNormBaseline'] = core_norm_baseline
        df.loc[i, 'HomoDefaultBaseline'] = default_norm_baseline(core.Baseline, norm='Homo')
        df.loc[i, f'HomoNorm{target}'] = core[target] / homo_run[target].mean()
        df.loc[i, f'HomoNorm{target}VsDefaultBaseline'] = core[target] / homo_run_default_norm[target].mean()

    # Get Homo Norm Target statistics for each run
    runs = df.groupby('RunName')
    for run_name in df.RunName.unique():
        run = runs.get_group(run_name)
        
        print('Run:', run_name)
        df.loc[run.index, f'HomoNorm{target}Sum'] = np.sum(run[f'HomoNorm{target}'])
        df.loc[run.index, f'HomoNorm{target}Product'] = np.prod(run[f'HomoNorm{target}'])
        df.loc[run.index, f'HomoNorm{target}Hmean'] = stats.hmean(run[f'HomoNorm{target}']) if not run[f'HomoNorm{target}'].isnull().all() else np.nan
        df.loc[run.index, f'HomoNorm{target}Var'] = np.var(run[f'HomoNorm{target}'])
        df.loc[run.index, f'HomoNorm{target}Std'] = np.std(run[f'HomoNorm{target}'])
        df.loc[run.index, f'HomoNorm{target}MSE'] = ((run[f'HomoNorm{target}'] - 1)**2).mean() # Squared error

        print(f'    HomoNorm{target}s (vs. NormBaseline {df.loc[run.index, f"HomoNormBaseline"].tolist()[0]})     :', run[f'HomoNorm{target}'].tolist())
        print(f'    HomoNorm{target}s (vs. DefaultBaseline {df.loc[run.index, f"HomoDefaultBaseline"].tolist()[0]}) :', run[f'HomoNorm{target}VsDefaultBaseline'].tolist())
        print(f'    NormBaseline ({df.loc[run.index, f"HomoNormBaseline"].tolist()[0]}) statistics:')
        print('        Sum    :', df.loc[run.index, f'HomoNorm{target}Sum'].tolist()[0])
        print('        Product:', df.loc[run.index, f'HomoNorm{target}Product'].tolist()[0])
        print('        HMean  :', df.loc[run.index, f'HomoNorm{target}Hmean'].tolist()[0])
        print('        Var    :', df.loc[run.index, f'HomoNorm{target}Var'].tolist()[0])
        print('        Std    :', df.loc[run.index, f'HomoNorm{target}Std'].tolist()[0])
        print('        MSE    :', df.loc[run.index, f'HomoNorm{target}MSE'].tolist()[0])

    return df


def add_single_norm_data(df, target, norm_baseline=None):
    """Compute the single-core-run normalized
    statistics for each core/run, after other stats
    have been calcuated.
    """
    df = df.reset_index(drop=True) 
    
    # Get Single Norm of Target
    # NOTE: (Baseline of a hawkeye_split is a hawkeye_simple)
    for i, core in df.iterrows():
        
        if norm_baseline is None:
            core_norm_baseline = default_norm_baseline(core.Baseline, norm='Single')
        else:
            core_norm_baseline = norm_baseline

        single_run = df[
            (df.Trace == core.Trace) &
            (df.Baseline == core_norm_baseline) &
            (df.NumCPUs == 1) &
            (df.LLCSetsPerCPU == core.LLCSetsPerCPU) &
            (df.HomogeneousMix == True)
        ]
        
        # For calculating FairUtility constraint on this particular core,
        # without introducing bias w.r.t a different normalization baseline.
        single_run_default_norm = df[
            (df.Trace == core.Trace) &
            (df.Baseline == default_norm_baseline(core.Baseline, norm='Single')) &
            (df.NumCPUs == 1) &
            (df.LLCSetsPerCPU == core.LLCSetsPerCPU) &
            (df.HomogeneousMix == True)
        ]
        
        
        df.loc[i, 'SingleNormBaseline'] = core_norm_baseline
        df.loc[i, 'SingleDefaultBaseline'] = default_norm_baseline(core.Baseline, norm='Single')
        df.loc[i, f'SingleNorm{target}'] = core[target] / single_run[target].mean()
        df.loc[i, f'SingleNorm{target}VsDefaultBaseline'] = core[target] / single_run_default_norm[target].mean()

    # Get Single Norm Target statistics for each run
    runs = df.groupby('RunName')
    for run_name in df.RunName.unique():
        run = runs.get_group(run_name)
        
        print('Run:', run_name)
        df.loc[run.index, f'SingleNorm{target}Sum'] = np.sum(run[f'SingleNorm{target}'])
        df.loc[run.index, f'SingleNorm{target}Product'] = np.prod(run[f'SingleNorm{target}'])
        df.loc[run.index, f'SingleNorm{target}Hmean'] = stats.hmean(run[f'SingleNorm{target}']) if not run[f'SingleNorm{target}'].isnull().all() else np.nan
        df.loc[run.index, f'SingleNorm{target}Var'] = np.var(run[f'SingleNorm{target}'])
        df.loc[run.index, f'SingleNorm{target}Std'] = np.std(run[f'SingleNorm{target}'])
        df.loc[run.index, f'SingleNorm{target}MSE'] = ((run[f'SingleNorm{target}'] - 1)**2).mean() # Squared error
        
        print(f'    SingleNorm{target}s (vs. NormBaseline {df.loc[run.index, f"SingleNormBaseline"].tolist()[0]}) :', run[f'SingleNorm{target}'].tolist())
        print(f'    SingleNorm{target}s (vs. DefaultBaseline {df.loc[run.index, f"SingleDefaultBaseline"].tolist()[0]}) :', run[f'SingleNorm{target}VsDefaultBaseline'].tolist())
        print(f'    NormBaseline ({df.loc[run.index, f"SingleNormBaseline"].tolist()[0]}) statistics:')
        print('        Sum    :', df.loc[run.index, f'SingleNorm{target}Sum'].tolist()[0])
        print('        Product:', df.loc[run.index, f'SingleNorm{target}Product'].tolist()[0])
        print('        HMean  :', df.loc[run.index, f'SingleNorm{target}Hmean'].tolist()[0])
        print('        Var    :', df.loc[run.index, f'SingleNorm{target}Var'].tolist()[0])
        print('        Std    :', df.loc[run.index, f'SingleNorm{target}Std'].tolist()[0])
        print('        MSE    :', df.loc[run.index, f'SingleNorm{target}MSE'].tolist()[0])

    return df


def build_run_statistics(results_dir, output_file, norm_baseline=None):
    """Build statistics for each run, per-core.
    """
    traces = {}
    for fn in os.listdir(results_dir):
        trace = fn.split('-hashed_perceptron-')[0]
        if trace not in traces:
            traces[trace] = {}

        repl_fn = fn.split('-hashed_perceptron-')[1].split('-')[4]
        llc_sets = int(fn.split('-hashed_perceptron-')[1].split('-')[6].replace('llc_sets', '').replace('.txt', ''))

        traces[trace][(repl_fn, llc_sets)] = os.path.join(results_dir, fn)

    columns = ['Trace', 'Baseline', 'CPU', 'NumCPUs', 'HawkeyeSplitAllocation',
               'LLCSets', 'LLCSetsPerCPU',
               'HomogeneousMix', 'MPKI', 'IPC', 'CPI', 'NumInstMillions',
               'LoadMisses', 'RFOMisses', 'RunName']
    
    run_df = pd.DataFrame(columns=columns)
    for trace in tqdm(traces, dynamic_ncols=True, unit='trace'):
        d = traces[trace]
        for baseline, llc_sets in d:
            run_df = run_df.append(compute_stats(
                d[(baseline, llc_sets)], 
                baseline_name=baseline
            ))

    # Get the Homo-normalized statistics for each run
    run_df = add_homo_norm_data(run_df, 'IPC', norm_baseline=norm_baseline)
    run_df = add_homo_norm_data(run_df, 'CPI', norm_baseline=norm_baseline)
    run_df = add_homo_norm_data(run_df, 'MPKI', norm_baseline=norm_baseline)

    # Get the Single-normalized statistics for each run
    run_df = add_single_norm_data(run_df, 'IPC', norm_baseline=norm_baseline)
    run_df = add_single_norm_data(run_df, 'CPI', norm_baseline=norm_baseline)
    run_df = add_single_norm_data(run_df, 'MPKI', norm_baseline=norm_baseline)

    run_df.to_csv(output_file, index=False)



def build_trace_statistics(run_stats_file):
    """Build statistics for each trace's fairness,
    using already-computed run statistics.
    """
    columns = ['Trace', 'Baseline', 'NumCPUs',
               'LLCSets', 'LLCSetsPerCPU',
               'MinIPC', 'MeanIPC', 'MaxIPC',
               'MinMPKI', 'MeanMPKI', 'MaxMPKI',
               'HomoNormMinIPC', 'HomoNormMeanIPC', 'HomoNormMaxIPC',
               'HomoNormMinMPKI', 'HomoNormMeanMPKI', 'HomoNormMaxMPKI']

    trace_df = pd.DataFrame(columns=columns)
    run_df = pd.read_csv(run_stats_file)

    # TODO - Clean up loop to do all three groups / uniques at once.
    t = run_df.groupby('Trace')
    for trace in run_df.Trace.unique():
        try:
            b = t.get_group(trace).groupby('Baseline')
        except KeyError:
            print(f'No runs match trace={trace}')
            continue
            
        for baseline in run_df.Baseline.unique():
            try:
                s = b.get_group(baseline).groupby('LLCSetsPerCPU')
            except KeyError:
                print(f'No runs match trace={trace}, baseline={baseline}')
                continue
                
            for llc_sets in run_df.LLCSets.unique():
                try:
                    c = s.get_group(llc_sets).groupby('NumCPUs')
                except KeyError:
                    print(f'No runs match trace={trace}, llc_sets_per_cpu={llc_sets}, baseline={baseline}')
                    continue
                    
                for n_cores in run_df.NumCPUs.unique():

                    try:
                        runs = c.get_group(n_cores)
                    except KeyError:
                        print(f'No runs match trace={trace}, llc_sets_per_cpu={llc_sets}, baseline={baseline}, n_cores={n_cores}')
                        continue

                    homo = runs[runs.HomogeneousMix == True]
                    homo_ipc = homo.IPC.mean() if not homo.empty else np.nan # Average homogeneous IPC (over the cores)
                    homo_mpki = homo.MPKI.mean() if not homo.empty else np.nan # Average homogeneous IPC (over the cores)

                    norm_ipcs = runs.IPC / homo_ipc    # IPCs normalized to homogeneous mix
                    norm_mpkis = runs.MPKI / homo_mpki # MPKIs normalized to homogeneous mix

                    trace_df.loc[len(trace_df.index)] = [
                        trace, baseline, n_cores, llc_sets * n_cores, llc_sets,
                        runs.IPC.min(), runs.IPC.mean(), runs.IPC.max(),
                        runs.MPKI.min(), runs.MPKI.mean(), runs.MPKI.max(),
                        norm_ipcs.min(), norm_ipcs.mean(), norm_ipcs.max(),
                        norm_mpkis.min(), norm_mpkis.mean(), norm_mpkis.max(),
                    ]

    trace_stats_file = run_stats_file.replace('.csv', '') + '_trace.csv'
    print(f'Saving dataframe to {trace_stats_file}...')
    trace_df.to_csv(trace_stats_file, index=False)


def eval_command():
    """Eval command
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('results_dir')
    parser.add_argument('--output-file', default=default_output_file)
    parser.add_argument('--norm-baseline', type=str, default=None)

    args = parser.parse_args(sys.argv[2:])

    print('=== Building run statistics... ===')
    if args.norm_baseline:
        norm_baseline = replacement_fns[replacement_names.index(args.norm_baseline)]
        print(f'Using normalization baseline {args.norm_baseline} ({norm_baseline})...')
    else:
        norm_baseline = None
        
    build_run_statistics(args.results_dir, args.output_file, norm_baseline=norm_baseline)

    print('=== [DEBUG] *NOT* building trace statistics... ===')
    #print('=== Building trace statistics... ===')
    #build_trace_statistics(args.output_file)




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
