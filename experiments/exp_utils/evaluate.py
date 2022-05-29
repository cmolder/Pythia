"""
Utility functions for evaluating Condor experiments on Pythia.

Authors: Quang Duong and Carson Molder
"""
from collections import defaultdict
import os
import glob
from typing import Union, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from exp_utils.file import ChampsimResultsDirectory, ChampsimResultsFile, ChampsimStatsFile

    
"""
Statistics helpers
"""
def get_run_statistics(file: ChampsimResultsFile, 
                   baseline_file: Optional[ChampsimResultsFile] = None) -> Union[dict, None]:
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
        'path', 
        'baseline_path'
    ]
    
    if baseline_file and file.prefetchers_match(baseline_file): # Don't compare baseline to itself.
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
    
    if baseline_file:
        b_data = baseline_file.read()
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
        
        if baseline_file:
            b_load_miss, b_rfo_miss, b_useful = (
               b_data[f'{level}_load_miss'], b_data[f'{level}_rfo_miss'], b_data[f'{level}_useful']
            )
            #b_total_miss = load_miss + rfo_miss + useful # = b_data[f'{level}_total_miss']
            b_total_miss = b_data[f'{level}_total_miss']
            b_mpki = b_total_miss / b_data['kilo_inst']
            assert np.isclose(b_data['kilo_inst'], pf_data['kilo_inst']), \
                (f'Traces {os.path.basename(baseline_file.path)}, {os.path.basename(file.path)} '
                 f'did not run for the same amount of instructions. '
                 f'({b_data["kilo_inst"]}K vs {pf_data["kilo_inst"]}K)')

        results[f'{level}_pref'] = file.get_prefetcher_at_level(level)
        results[f'{level}_pref_degree'] = file.get_prefetcher_degree_at_level(level)
        results[f'{level}_accuracy'] = 100.0 if (useful + useless == 0) else useful / (useful + useless) * 100.
        results[f'{level}_coverage'] = np.nan if (total_miss == 0 or baseline_file is None) else ((b_load_miss + b_rfo_miss) - (load_miss + rfo_miss)) / (b_load_miss + b_rfo_miss) * 100
        results[f'{level}_mpki'] = pf_mpki
        results[f'{level}_mpki_reduction'] = np.nan if (baseline_file is None) else (b_mpki - pf_mpki) / b_mpki * 100.
                
        
    results['dram_bw_epochs'] = dram_bw_epochs
    results['dram_bw_reduction'] = np.nan if (baseline_file is None) else (b_dram_bw_epochs - dram_bw_epochs) / b_dram_bw_epochs * 100.
    results['ipc'] = ipc
    results['ipc_improvement'] = np.nan if (baseline_file is None) else (ipc - b_ipc) / b_ipc * 100.  
    results['seed'] = pf_data['seed']
    results['path'] = file.path
    results['baseline_path'] = baseline_file.path 
    assert all([k in stats_columns for k in results.keys()]), f'Columns missing for row in {file.path}'
    
    return results


def get_pc_statistics(file: ChampsimStatsFile) -> list[dict]:
    """Get per-PC statistics from a Champsim pc_pref_stats file.
    """
    pc_columns = [
        'pc', 'full_trace', 'trace', 'simpoint', 
        'pref', 'pref_degree', 'num_useful', 
        'num_useless', 'accuracy',
    ]
    

    # Get statistics
    pc_out = []
    for pc in file.read().keys(): 
        row = {k : np.nan for k in pc_columns}
        row['pc'] = pc
        row['full_trace'] = file.full_trace
        row['trace'] = file.trace
        row['simpoint'] = file.simpoint
        row['pref'] = file.get_prefetcher_at_level(file.level)
        row['pref_degree'] = file.get_prefetcher_degree_at_level(file.level)
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
def generate_run_csv(results_dir: str, 
                     output_file: str, 
                     dry_run: bool = False) -> None:
    """Generate cumulative statistics for each run.
    """
    directory = ChampsimResultsDirectory(results_dir)
    stats = []
    
    # Build stats table
    # TODO : Break this out into a function or write some code to cleanly
    #        and orderly loop through the files in the Directory.
    with tqdm(total=len(directory), dynamic_ncols=True, unit='trace') as pbar:
        for tr in directory.files.keys():
            for seed in directory[tr].keys():
                assert directory.get_baseline(tr, seed) is not None, \
                    f'Could not find baseline for trace {tr}, seed {seed}'
                for pf in directory[tr][seed].keys():
                    for v in directory[tr][seed][pf].keys():
                        pbar.update(1)
                        row = get_run_statistics(
                            directory[tr][seed][pf][v], 
                            baseline_file=directory.get_baseline(tr, seed)
                        )         

                        if row is not None:
                            stats.append(row) # Append row to list, if it isn't None
                    
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
    directory = ChampsimResultsDirectory(results_dir)
    prefetchers = set()
    best_degree = defaultdict(dict)

    # Build best degree dictionary
    # Compute the best_degree for each prefetcher on each trace.
    # TODO : Break this out into a function or write some code to cleanly
    #        and orderly loop through the files in the Directory.
    with tqdm(total=len(directory), dynamic_ncols=True, unit='trace') as pbar:
        for tr in directory.files.keys():
            seed = list(directory[tr].keys()[0]) # Just consider one seed, whichever one comes first in the list.
            assert directory.get_baseline(tr, seed) is not None, \
                f'Could not find baseline for trace {tr}, seed {seed}'
            for pf in directory[tr][seed].keys():
                if pf == ('no', 'no', 'no'):
                    pbar.update(1)
                    continue
                prefetchers.add(pf)
                scores = defaultdict(lambda : float('-inf'))

                for v in directory[tr][seed][pf].keys():
                    pbar.update(1)
                    # TODO: Consider other parts of the variant?
                    l2d, lld = v[0][1], v[1][1] # TODO: Do this without the hack.
                    row = get_run_statistics(directory[tr][seed][pf][v])
                    if row is not None:
                        scores[','.join(str(l2d), str(lld))] = row[metric]
                        #print('[DEBUG]', tr, seed, pf, l2d, lld, row[metric])

                best_degree[tr][pf] = max(scores, key=scores.get)
     
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
    directory = ChampsimStatsDirectory(os.path.join(results_dir, 'pc_pref_stats'))
    stats = []
    
    # Build stats table
    # TODO : Break this out into a function or write some code to cleanly
    #        and orderly loop through the files in the Directory.
    with tqdm(total=len(directory), dynamic_ncols=True, unit='trace') as pbar:
        for tr in directory.files.keys():
            seed = list(directory[tr].keys()[0]) # Just consider one seed, whichever one comes first in the list.
            for pf in directory[tr][seed].keys():
                if pf == ('no', 'no', 'no'):
                    pbar.update(1)
                    continue
                    
                for v in traces[tr][seed][pf].keys():
                    pbar.update(1)
                    rows = get_pc_statistics(
                        traces[tr][seed][pf][v]
                    )
                    stats.extend(rows)

    # Save statistics table
    stats = pd.DataFrame(stats)
    if not dry_run:
        print(f'Saving per-PC {level} prefetch statistics table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)
