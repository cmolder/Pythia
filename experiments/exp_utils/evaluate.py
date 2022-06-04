"""
Utility functions for evaluating Condor experiments on Pythia.

Authors: Quang Duong and Carson Molder
"""
from collections import defaultdict
import os
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from exp_utils.file import (ChampsimResultsDirectory, ChampsimStatsDirectory,
                            ChampsimResultsFile, ChampsimStatsFile)


def get_run_statistics(
        file: ChampsimResultsFile,
        baseline_file: Optional[ChampsimResultsFile] = None) -> Optional[dict]:
    """Get cumulative statistics from a ChampSim output / results file.
    """
    stats_columns = [
        'full_trace', 'trace', 'simpoint', 'L1D_pref', 'L1D_pref_degree',
        'L1D_accuracy', 'L1D_coverage', 'L1D_mpki', 'L1D_mpki_reduction',
        'L2C_pref', 'L2C_pref_degree', 'L2C_accuracy', 'L2C_coverage',
        'L2C_mpki', 'L2C_mpki_reduction', 'LLC_pref', 'LLC_pref_degree',
        'LLC_accuracy', 'LLC_coverage', 'LLC_mpki', 'LLC_mpki_reduction',
        'dram_bw_epochs', 'dram_bw_reduction', 'ipc', 'ipc_improvement',
        'pythia_level_threshold', 'pythia_high_conf_prefetches',
        'pythia_low_conf_prefetches', 'pythia_features', 'seed', 'path',
        'baseline_path'
    ]

    # Don't compare baseline to itself.
    if baseline_file and file.prefetchers_match(baseline_file):
        return None

    # Get statistics
    pf_data = file.read()
    if pf_data == None:
        print(f'Warning: Missing data for {file.path}')
        return None

    results = {k: np.nan for k in stats_columns}
    results['full_trace'] = file.full_trace
    results['trace'] = file.trace
    results['simpoint'] = file.simpoint
    results['pythia_level_threshold'] = pf_data['pythia_level_threshold']
    results['pythia_features'] = pf_data['pythia_features']
    results['pythia_high_conf_prefetches'] = pf_data[
        'pythia_high_conf_prefetches']
    results['pythia_low_conf_prefetches'] = pf_data[
        'pythia_low_conf_prefetches']

    if baseline_file:
        b_data = baseline_file.read()
        b_ipc, b_dram_bw_epochs, b_kilo_inst = (b_data['ipc'],
                                                b_data['dram_bw_epochs'],
                                                b_data['kilo_inst'])

    ipc, dram_bw_epochs = (pf_data['ipc'], pf_data['dram_bw_epochs'])

    # Get per-level statistics
    for level in ['L1D', 'L2C', 'LLC']:
        iss_prefetches, useful, useless, load_miss, rfo_miss = (
            pf_data[f'{level}_issued_prefetches'], pf_data[f'{level}_useful'],
            pf_data[f'{level}_useless'], pf_data[f'{level}_load_miss'],
            pf_data[f'{level}_rfo_miss'])

        total_miss = load_miss + rfo_miss + useful
        pf_mpki = (load_miss + rfo_miss) / pf_data['kilo_inst']

        if baseline_file:
            b_load_miss, b_rfo_miss, b_useful = (b_data[f'{level}_load_miss'],
                                                 b_data[f'{level}_rfo_miss'],
                                                 b_data[f'{level}_useful'])
            #b_total_miss = load_miss + rfo_miss + useful # = b_data[f'{level}_total_miss']
            b_total_miss = b_data[f'{level}_total_miss']
            b_mpki = b_total_miss / b_data['kilo_inst']
            assert np.isclose(b_data['kilo_inst'], pf_data['kilo_inst']), \
                (f'Traces {os.path.basename(baseline_file.path)}, '
                 f'{os.path.basename(file.path)} '
                 f'did not run for the same amount of instructions. '
                 f'({b_data["kilo_inst"]}K vs {pf_data["kilo_inst"]}K)')

        results[f'{level}_pref'] = file.get_prefetcher_at_level(level)
        results[f'{level}_pref_degree'] = file.get_prefetcher_degree_at_level(
            level)
        results[f'{level}_accuracy'] = 100.0 if (
            useful + useless == 0) else useful / (useful + useless) * 100.
        results[f'{level}_coverage'] = np.nan if (
            total_miss == 0 or baseline_file is None) else (
                (b_load_miss + b_rfo_miss) -
                (load_miss + rfo_miss)) / (b_load_miss + b_rfo_miss) * 100
        results[f'{level}_mpki'] = pf_mpki
        results[f'{level}_mpki_reduction'] = np.nan if (
            baseline_file is None) else (b_mpki - pf_mpki) / b_mpki * 100.

    # Get cumulative statistics
    results['dram_bw_epochs'] = dram_bw_epochs
    results['dram_bw_reduction'] = np.nan if (baseline_file is None) else (
        b_dram_bw_epochs - dram_bw_epochs) / b_dram_bw_epochs * 100.
    results['ipc'] = ipc
    results['ipc_improvement'] = np.nan if (
        baseline_file is None) else (ipc - b_ipc) / b_ipc * 100.
    results['seed'] = pf_data['seed']
    results['path'] = file.path
    results['baseline_path'] = baseline_file.path
    assert all([k in stats_columns for k in results.keys()
                ]), f'Columns missing for row in {file.path}'
    return results


def get_pc_statistics(file: ChampsimStatsFile) -> list[dict]:
    """Get per-PC statistics from a Champsim pc_pref_stats file.
    """
    pc_columns = [
        'pc',
        'full_trace',
        'trace',
        'simpoint',
        'pref',
        'pref_degree',
        'num_useful',
        'num_useless',
        'accuracy',
    ]

    # Get statistics
    pc_out = []
    pc_data = file.read()
    for pc in pc_data:
        row = {k: np.nan for k in pc_columns}
        row['pc'] = pc
        row['full_trace'] = file.full_trace
        row['trace'] = file.trace
        row['simpoint'] = file.simpoint
        row['pref'] = file.get_prefetcher_at_level(file.level)
        row['pref_degree'] = file.get_prefetcher_degree_at_level(file.level)
        row['num_useful'] = pc_data[pc]['useful']
        row['num_useless'] = pc_data[pc]['useless']
        row['accuracy'] = (100.0 if (pc_data[pc]['useful'] +
                                     pc_data[pc]['useless'] == 0) else
                           (pc_data[pc]['useful'] /
                            (pc_data[pc]['useful'] + pc_data[pc]['useless'])))
        pc_out.append(row)

    return pc_out


def generate_run_csv(results_dir: str,
                     output_file: str,
                     dry_run: bool = False) -> None:
    """Generate cumulative statistics for each run.
    """
    directory = ChampsimResultsDirectory(results_dir)
    stats = []

    # Build stats table
    # TODO: Can we wrap this routine by passing a callable? That way, we can
    #       reuse code between the CSV file creators.
    for file in tqdm(directory, dynamic_ncols=True, unit='file'):
        assert directory.get_baseline(
            file.full_trace, file.champsim_seed) is not None, (
                f'Could not find baseline for trace {file.full_trace}, '
                f'seed {file.champsim_seed}')
        baseline_file = directory.get_baseline(file.full_trace,
                                               file.champsim_seed)
        row = get_run_statistics(file, baseline_file=baseline_file)

        if row is not None:
            stats.append(row)  # Append row to list, if it isn't None

    # Build and save statistics table
    stats = pd.DataFrame(stats)
    stats.sort_values(by=[
        'full_trace', 'L1D_pref', 'L2C_pref', 'LLC_pref', 'L2C_pref_degree',
        'LLC_pref_degree', 'pythia_level_threshold', 'pythia_features'
    ],
                      inplace=True)
    if not dry_run:
        print(f'Saving statistics table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)


def generate_best_degree_csv(results_dir: str,
                             output_file: str,
                             metric: str = 'ipc',
                             dry_run: bool = False) -> None:
    """Generate the best degree for each prefetcher on each run.
    """
    directory = ChampsimResultsDirectory(results_dir)
    prefetchers = set()
    scores = defaultdict(lambda: defaultdict(dict))

    # Build scores dictionary
    # TODO: Can we wrap this routine by passing a callable? That way, we can
    #       reuse code between the CSV file creators.
    for file in tqdm(directory, dynamic_ncols=True, unit='file'):
        tr = file.full_trace
        pf = file.get_all_prefetchers()

        # Skip baseline
        if pf == ChampsimResultsFile.get_baseline_prefetchers():
            continue

        prefetchers.add(pf)
        row = get_run_statistics(file)

        # TODO : Consider other parts of the variant, seed (currently
        #        overrides with the last-seen file)
        if row is not None:
            scores[tr][pf][','.join(file.l2c_prefetcher_degeree,
                                    file.llc_prefetcher_degree)] = row[metric]

    # Loop through trace+prefetchers and get best score on each.
    best_degree = defaultdict(dict)
    for tr in scores:
        for pf in scores[tr]:
            best_degree[tr][pf] = max(scores[tr][pf], key=scores[tr][pf].get)

    # Turn best_degree dictionary into a table
    df = pd.DataFrame(columns=['Trace'] + list(prefetchers))
    for tr in best_degree.keys():
        row = best_degree[tr]
        row['Trace'] = tr
        df = df.append(best_degree[tr], ignore_index=True)
    df.sort_values('Trace', inplace=True)

    # Save best degree table
    if not dry_run:
        print(f'Saving best degree table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = df.to_csv(output_file, index=False)


def generate_pc_csv(results_dir: str,
                    output_file: str,
                    level: str = 'llc',
                    dry_run: bool = False) -> None:
    """Generate statistics on each PC for each prefetcher on each run.
    """
    directory = ChampsimStatsDirectory(
        os.path.join(results_dir, 'pc_pref_stats'))
    stats = []

    # Build stats table
    # TODO: Try to filter on or explicitly track seed / variants if we can, to avoid
    #       excessive repeats. But this might not be necessary.
    # TODO: Can we wrap this routine by passing a callable? That way, we can
    #       reuse code between the CSV file creators.
    for file in tqdm(directory, dynamic_ncols=True, unit='file'):
        if (file.get_all_prefetchers() ==
                ChampsimStatsFile.get_baseline_prefetchers()):
            continue
        rows = get_pc_statistics(file)
        stats.extend(rows)

    # Save statistics table
    stats = pd.DataFrame(stats)
    stats.sort_values(by=['full_trace', 'pref', 'pref_degree'], inplace=True)
    if not dry_run:
        print(f'Saving per-PC {level} prefetch statistics table '
              f'to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)
