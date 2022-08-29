"""Utility classes and functions for working with PC traces.

Author: Carson Molder
"""

import glob
import gzip
from multiprocessing import get_context
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


metrics = ['num_useful', 'marginal_useful', 'accuracy', 'coverage']


def get_pc_trace_file(trace, metric, level='llc'):
    """Return name of a pc trace file.

    TODO: Docstring parameters/returns
    """
    return f'{trace}_{metric}_{level}_pc_trace.txt'

def _format_prefetcher(pref):
    """TODO: Docstring
    """
    pref = pref.replace('spp_dev2', 'sppdev2')
    pref = pref.replace('_', ',')
    pref = pref.replace('sppdev2', 'spp_dev2')
    return pref


def _format_degree(deg):
    """TODO: Docstring
    """
    deg = eval(deg)
    deg = ','.join([str(d) if d is not None else 'na' for d in deg])
    return deg


def get_llc_prefetcher_from_path(path):
    """TODO: Docstring
    """
    llc_pref = os.path.basename(path).split('-')[4]
    return llc_pref


def get_best_prefetcher_degree(pc_data, metric, benchmark):
    """TODO: Docstring
    """
    # Build dataset
    assert metric in metrics, (
        f'Metric {metric} not in supported metrics {metrics}')
    bk_data = pc_data[pc_data.full_trace == benchmark]
    bk_data = bk_data[bk_data.pref != 'no']
    best_prefetcher = {}
    best_degree = {}

    for pc in bk_data.pc.unique():
        pc_data = bk_data[bk_data.pc == pc]

        # Accuracy:
        # 1. Pick the prefetcher with the highest accuracy
        # 2. On a tie, pick the prefetcher with the most useful prefetches.
        # 3. If still tied, pick one at random.
        if metric == 'accuracy':
            best_pf = pc_data[pc_data.accuracy == pc_data.accuracy.max()]
            best_pf = best_pf[best_pf.num_useful == best_pf.num_useful.max()]
            best_pf = best_pf.sample(n = 1)

        # Num Useful:
        # 1. Pick the prefetcher with the most useful prefetches
        # 2. On a tie, pick the prefetcher with the highest accuracy
        # 3. If still tied, pick one at random
        elif metric == 'num_useful':
            best_pf = pc_data[pc_data.num_useful == pc_data.num_useful.max()]
            best_pf = best_pf[best_pf.accuracy == best_pf.accuracy.max()]
            best_pf = best_pf.sample(n = 1)

        # Marginal Useful:
        # 1. Pick the prefetcher with the largest (useful - useless) prefetches
        # 2. On a tie, pick the prefetcher with most useful prefetches
        # 3. If still tied, pick one at random
        elif metric == 'marginal_useful':
            pc_data['marginal_useful'] = (
                pc_data.num_useful - pc_data.num_useless)
            best_pf = pc_data[
                pc_data.marginal_useful == pc_data.marginal_useful.max()]
            best_pf = best_pf[best_pf.num_useful == best_pf.num_useful.max()]
            best_pf = best_pf[best_pf.accuracy == best_pf.accuracy.max()]
            best_pf = best_pf.sample(n = 1)

        elif metric == 'coverage':
            pc_data.coverage = pc_data.coverage.fillna(-np.inf)
            best_pf = pc_data[pc_data.coverage == pc_data.coverage.max()]
            best_pf = best_pf[best_pf.num_useful == best_pf.num_useful.max()]
            best_pf = best_pf[best_pf.accuracy == best_pf.accuracy.max()]
            best_pf = best_pf.sample(n = 1)

        best_prefetcher[pc] = best_pf.pref.item()
        best_degree[pc] = best_pf.pref_degree.item()

    return best_prefetcher, best_degree



# Online PC traces

# Idea: Run and train the best prefetcher for each PC (online)
# Form: `pc pref1,pref2,...,prefn deg1,deg2,...,degn`
# For: multi_pc_trace
def build_online_pc_traces(pc_stats_file, output_dir, metric,
                           level='llc', dry_run=False):
    """Build an online PC trace, of the form:

    `pc pref1,pref2,...,prefn deg1,deg2,...,degn`

    Where pref1,... is the best-performing prefetcher (hybrid), on that
    PC, under the metric.

    For evaluation of the prefetcher choices online (via multi_pc_trace).

    TODO: Docstring parameters/returns
    """
    assert metric in metrics, (
        f'Metric {metric} not in supported metrics {metrics}')
    data = pd.read_csv(pc_stats_file)
    benchmarks = sorted(data.full_trace.unique().tolist())

    print('Building online PC traces for...')
    print('    pc_stats_file:', pc_stats_file)
    print('    output_dir   :', output_dir)
    print('    metric       :', metric)
    print('    benchmarks   :', ', '.join(benchmarks))

    for benchmark in tqdm(benchmarks, dynamic_ncols=True, unit='simpoint'):
        best_prefetcher, best_degree = get_best_prefetcher_degree(
            data, metric, benchmark)

        # Write PC traces
        # Lines are of the form `pc pref1,pref2,...,prefn deg1,deg2,...,degn`
        output_file = os.path.join(
            output_dir, get_pc_trace_file(benchmark, metric, level))
        if not dry_run:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w') as pc_trace_f:
                for pc, best_pref in best_prefetcher.items():
                    print(
                        f'{pc} {_format_prefetcher(best_pref)} '
                        f'{_format_degree(best_degree[pc])}', file=pc_trace_f)



# Offline PC traces

# Idea: Replay the prefetches of the best prefetcher for each PC (offline)
# Form: `instr_id call_num pc addr1,addr2,...,addrn deg1,deg,...,degn`
# For: from_file
def _get_prefetch_traces(pref_traces_dir, benchmark):
    """Get a list of prefetcher traces, filtering out
    any already-generated offline PC traces.

    TODO: Docstring parameters/returns
    """
    pc_traces = glob.glob(
        os.path.join(pref_traces_dir, benchmark + '*pc_trace_*.gz'))
    traces = glob.glob(
        os.path.join(pref_traces_dir, benchmark + '*.gz'))
    return list(set(traces) - set(pc_traces))


def _dump_offline_pc_trace(output_trace_file, instr_id_lines):
    """TODO: Docstring
    """
    BUF_MAX = 10000
    buf = ''
    with open(output_trace_file, 'wb') as f:
        for instr_id in instr_id_lines:
            buf += instr_id_lines[instr_id]
            if len(buf) > BUF_MAX:
                f.write(buf.encode())
                buf = ''

        # Wrte any remaining data
        f.write(buf.encode())

def _process_offline_pc_benchmark(inputs):
    """TODO: Docstring
    """
    data, metric, pref_traces_dir, level, benchmark = inputs
    #print(f'{benchmark:20} ({i}/{len(benchmarks)})')

    best_prefetcher, best_degree = get_best_prefetcher_degree(
        data, metric, benchmark)

    traces      = _get_prefetch_traces(pref_traces_dir, benchmark)

    # LLC prefetchers
    prefetchers = [get_llc_prefetcher_from_path(tr) for tr in traces]

    # Indexed by instruction ID (will overwrite if it sees one twice)
    # TODO: Dump output to file every now and then (in append mode), instead
    #       of keeping the whole dict in memory, to save memory.
    output = {}



    # Read all the traces into memory, building the combined trace
    # instruction-by-instruction
    for j, (t, p) in enumerate(zip(traces, prefetchers)):
        print(f'{benchmark:20}     (trace {j+1:2}/{len(traces)}) {p:20}')
        with gzip.open(t) as f:
            for line in f:
                line = str(line, 'utf-8')
                instr_id, _, pc, _, _ = line.split()
                if not (pc in best_prefetcher and pc in best_degree):
                    continue # No prefetch (I think)

                best_p = (
                    _format_prefetcher(best_prefetcher[pc])
                    + '_' + _format_degree(best_degree[pc]))
                if p == best_p:
                    output[instr_id] = line

    # Write offline combined trace
    output_trace_path = os.path.join(
        pref_traces_dir, get_pref_trace_file(benchmark, metric, level)
    )

    # Dump offline cominbed trace
    print(
        f'{benchmark:20}     '
        f'Building done, saving {os.path.basename(output_trace_path)}...')
    _dump_offline_pc_trace(output_trace_path, output)
    print(f'{benchmark:20}     Saving done')


def get_pref_trace_file(full_trace, metric, level='llc'):
    """TODO: Docstring
    """
    return f'{full_trace}_{metric}_pc_trace_{level}.gz'


def build_offline_pc_traces(pref_traces_dir, pc_stats_file, metric,
                            level='llc', num_threads=1, dry_run=False):
    """Build an offline PC trace, of the form:

    `instr_id call_num pc addr1,addr2,...,addrn deg1,deg,...,degn`

    Where addr1,... come from the best-performing prefetcher for that PC,
    under the metric, for that particular instruction.

    For evaluation of the prefetcher choices offline (via from_file).

    TODO: Docstring parameters/returns
    """
    assert metric in metrics, (
        f'Metric {metric} not in supported metrics {metrics}')
    assert level == 'llc', 'Only support LLC for now'
    data = pd.read_csv(pc_stats_file)

    benchmarks = sorted(data.full_trace.unique().tolist())

    print('Building offline PC traces for...')
    print('    pc_stats_file  :', pc_stats_file)
    print('    pref_traces_dir:', pref_traces_dir)
    print('    metric         :', metric)
    print('    benchmarks     :', ', '.join(benchmarks))
    print('    # of threads   :', num_threads)

    #all_traces =  glob.glob(os.path.join(pref_traces_dir, '*.gz'))

    #inputs = [(data, metric, b) for b in benchmarks]
    if not dry_run:
        with get_context('spawn').Pool(processes=num_threads) as pool:
            # inputs = [ # [DEBUG]
            #     (data, metric, pref_traces_dir, level, b)
            #     for b in ['bwaves_98B']
            # ]
            inputs = [(data, metric, pref_traces_dir, level, b)
                      for b in benchmarks]
            pool.map(_process_offline_pc_benchmark, inputs)
