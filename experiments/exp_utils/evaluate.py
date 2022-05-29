"""
Utility functions for evaluating Condor experiments on Pythia.

Authors: Quang Duong and Carson Molder
"""
from collections import defaultdict
import os
import glob
from typing import Union

import pandas as pd
import numpy as np
from tqdm import tqdm
from exp_utils.file import ChampsimResultsFile, ChampsimStatsFile

    
"""
Statistics helpers
"""
def get_statistics(path: str, baseline_path: str = None) -> Union[dict, None]:
    """Get cumulative statistics from a ChampSim output / results file.
    """
    stats_columns = [
        'full_trace', 'trace', 'simpoint', 
        'L1D_pref', 'L1D_pref_degree', 'L1D_accuracy', 
        'L1D_coverage', 'L1D_mpki', 'L1D_mpki_reduction',
        'L2C_pref', 'L2C_pref_degree', 'L2C_accuracy', 
        'L2C_coverage', 'L2C_mpki', 'L2C_mpki_reduction',
        'LLC_pref', 'LLC_pref_degree', 'LLC_accuracy', 
        'LLC_coverage', 'LLC_mpki', 'LLC_mpki_reduction',
        'dram_bw_epochs', 'dram_bw_reduction', 
        'ipc', 'ipc_improvement',
        'pythia_level_threshold', 
        'pythia_high_conf_prefetches', 
        'pythia_low_conf_prefetches',
        'pythia_features',
        'seed',
        'path', 'baseline_path'
    ]
    
    file = ChampsimResultsFile(path)
    b_file = ChampsimResultsFile(baseline_path) if baseline_path else None
        
    if b_file and file.prefetchers_match(b_file): # Don't compare baseline to itself.
        return None 
    
    # Get statistics
    pf_data = file.read()
    if pf_data == None:
        print(f'Warning: Missing data for {file.path}')
        return None
    
    results = {k : np.nan for k in stats_columns}
    results['full_trace'] = file.full_trace
    results['trace'] = file.trace
    results['simpoint'] = file.simpoint
    results['pythia_level_threshold'] = pf_data['pythia_level_threshold']
    results['pythia_features'] = pf_data['pythia_features']
    results['pythia_high_conf_prefetches'] = pf_data['pythia_high_conf_prefetches']
    results['pythia_low_conf_prefetches'] = pf_data['pythia_low_conf_prefetches']
    
    if baseline_path:
        b_data = b_file.read()
        b_ipc, b_dram_bw_epochs, b_kilo_inst = (
            b_data['ipc'], b_data['dram_bw_epochs'], b_data['kilo_inst']
        )
    
    ipc, dram_bw_epochs = (
        pf_data['ipc'], pf_data['dram_bw_epochs']
    )
    
    
    for level in ['L1D', 'L2C', 'LLC']:
        iss_prefetches, useful, useless, load_miss, rfo_miss = (
            pf_data[f'{level}_issued_prefetches'],
            pf_data[f'{level}_useful'],  pf_data[f'{level}_useless'],  
            pf_data[f'{level}_load_miss'], pf_data[f'{level}_rfo_miss']
        )
        
        total_miss = load_miss + rfo_miss + useful
        pf_mpki = (load_miss + rfo_miss) / pf_data['kilo_inst']
        
        if baseline_path:
            b_load_miss, b_rfo_miss, b_useful = (
               b_data[f'{level}_load_miss'], b_data[f'{level}_rfo_miss'], b_data[f'{level}_useful']
            )
            #b_total_miss = load_miss + rfo_miss + useful # = b_data[f'{level}_total_miss']
            b_total_miss = b_data[f'{level}_total_miss']
            b_mpki = b_total_miss / b_data['kilo_inst']
            assert np.isclose(b_data['kilo_inst'], pf_data['kilo_inst']), f'Traces {os.path.basename(path)}, {os.path.basename(path)} did not run for the same amount of instructions. ({b_data["kilo_inst"]}K vs {pf_data["kilo_inst"]}K)'

        results[f'{level}_pref'] = file.prefetcher_at_level(level)
        results[f'{level}_pref_degree'] = file.prefetcher_degree_at_level(level)
        results[f'{level}_accuracy'] = 100.0 if (useful + useless == 0) else useful / (useful + useless) * 100.
        results[f'{level}_coverage'] = np.nan if (total_miss == 0 or baseline_path is None) else ((b_load_miss + b_rfo_miss) - (load_miss + rfo_miss)) / (b_load_miss + b_rfo_miss) * 100
        results[f'{level}_mpki'] = pf_mpki
        results[f'{level}_mpki_reduction'] = np.nan if (baseline_path is None) else (b_mpki - pf_mpki) / b_mpki * 100.
                
        
    results['dram_bw_epochs'] = dram_bw_epochs
    results['dram_bw_reduction'] = np.nan if (baseline_path is None) else (b_dram_bw_epochs - dram_bw_epochs) / b_dram_bw_epochs * 100.
    results['ipc'] = ipc
    results['ipc_improvement'] = np.nan if (baseline_path is None) else (ipc - b_ipc) / b_ipc * 100.  
    results['seed'] = pf_data['seed']
    results['path'] = path
    results['baseline_path'] = baseline_path 
    assert all([k in stats_columns for k in results.keys()]), f'Columns missing for row in {path}'
    
    return results


def get_pc_statistics(path: str) -> list[dict]:
    """Get per-PC statistics from a Champsim pc_pref_stats file.
    """
    pc_columns = [
        'pc', 'full_trace', 'trace', 'simpoint', 
        'pref', 'pref_degree', 'num_useful', 
        'num_useless', 'accuracy',
    ]
    
    file = ChampsimStatsFile(path)

    # Get statistics
    pc_out = []
    for pc in file.read().keys(): 
        row = {k : np.nan for k in pc_columns}
        row['pc'] = pc
        row['full_trace'] = file.full_trace
        row['trace'] = file.trace
        row['simpoint'] = file.simpoint
        row['pref'] = file.prefetcher_at_level(file.level)
        row['pref_degree'] = file.prefetcher_degree_at_level(file.level)
        row['num_useful'] = pc_data[pc]['useful']
        row['num_useless'] =  pc_data[pc]['useless']
        row['accuracy'] = (100.0 if (pc_data[pc]['useful'] + pc_data[pc]['useless'] == 0) 
                           else (pc_data[pc]['useful'] 
                                 / (pc_data[pc]['useful'] + pc_data[pc]['useless'])))
        pc_out.append(row)
        
    return pc_out
    

"""
CSV file creators
"""
def generate_run_csv(results_dir: str, output_file: str, dry_run: bool = False):
    """Generate cumulative statistics for each run.
    """
    traces = defaultdict(lambda : defaultdict(dict))
    paths = [ChampsimResultsFile(p) for p in glob.glob(os.path.join(results_dir, '*.txt'))]
    n_paths = len(paths)
    
    # Build trace paths
    # TODO : Breakout into a Traceholder class.
    for path in paths:
        (traces[path.full_trace]
               [(path.l1_prefetcher, path.l2_prefetcher, path.llc_prefetcher)]
               [(path.l2_prefetcher_degree, path.llc_prefetcher_degree,
                path.pythia_level_threshold, path.pythia_features,
                path.champsim_seed)]) = path.path

        
    # Build statistics table
    stats = []
    with tqdm(total=n_paths, dynamic_ncols=True, unit='trace') as pbar:
        for tr in traces:
            assert ('no', 'no', 'no') in traces[tr].keys(), f'Could not find baseline ("no", "no", "no") run for trace {tr}'
            for pf in traces[tr]:
                for d in traces[tr][pf]:
                    _, _, _, _, seed = d
                    pbar.update(1)

                    row = get_statistics(traces[tr][pf][d], baseline_path=traces[tr][('no', 'no', 'no')][((None,), (None,), None, None, seed)])         
                    if row is None: # Filter missing rows
                        continue

                    
                    stats.append(row)
                    
    # Build and save statistics table
    stats = pd.DataFrame(stats)
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
    traces = defaultdict(lambda : defaultdict(dict))
    prefetchers = set()
    best_degree = defaultdict(dict)
    paths = [ResultsPath(p) for p in glob.glob(os.path.join(results_dir, '*.txt'))]
    n_paths = len(paths)

    # Build trace paths
    # TODO : Breakout into a Traceholder class.
    for path in paths:
        (traces[path.full_trace] # TODO: Index on PLT, features, seed
               [(path.l1_prefetcher, path.l2_prefetcher, path.llc_prefetcher)]
               [(path.l2_prefetcher_degree, path.llc_prefetcher_degree)]) = path.path

    # Build best degree dictionary
    # - Compute the best_degree for each prefetcher on each trace.
    with tqdm(total=n_paths, dynamic_ncols=True, unit='trace') as pbar:
        for tr in traces.keys():
            for pf in traces[tr].keys():
                if pf == 'no':
                    pbar.update(1)
                    continue
                prefetchers.add(pf)
                    
                scores = defaultdict(lambda : float('-inf'))
                for d in traces[tr][pf].keys():
                    pbar.update(1)    
                    row = get_statistics(traces[tr][pf][d], baseline_path=None)
                    #print('[DEBUG]', tr, pf, d, row[metric])
                    if row is None:
                        continue
                    scores[d] = row[metric]

                best_degree[tr][pf] = max(scores, key=scores.get)
                #print('[DEBUG]', tr, pf, 'best_degree =', best_degree[tr][pf])
    
    # Turn best_degree dictionary into a table
    df = pd.DataFrame(columns=['Trace'] + list(prefetchers))
    for tr in best_degree.keys():
        row = best_degree[tr]
        row['Trace'] = tr
        df = df.append(best_degree[tr], ignore_index=True)
    
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
    traces = defaultdict(lambda : defaultdict(dict))
    paths = [ChampsimStatsFile(p) 
             for p in glob.glob(
                 os.path.join(results_dir, 'pc_pref_stats', f'*_{level}.txt'))]
    
    n_paths = len(paths)
    
    # Build trace paths
    # TODO : Breakout into a Traceholder class.
    for path in paths:
        (traces[path.full_trace] # TODO: Index on PLT, features, seed
               [(path.l1_prefetcher, path.l2_prefetcher, 
                 path.llc_prefetcher)]
               [(path.l2_prefetcher_degree, 
                 path.llc_prefetcher_degree)]) = path.path
       
    # Build statistics table
    stats = []
    with tqdm(total=n_paths, dynamic_ncols=True, unit='trace') as pbar:
        for tr in traces.keys():
            for pf in traces[tr].keys():
                if pf == (('no',), ('no',), ('no',)):
                    pbar.update(1)
                    continue
                
                for d in traces[tr][pf].keys():
                    pbar.update(1)
                    rows = get_pc_statistics(traces[tr][pf][d])
                    stats.extend(rows)

    # Save statistics table
    stats = pd.DataFrame(stats)
    if not dry_run:
        print(f'Saving per-PC {level} prefetch statistics table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)
