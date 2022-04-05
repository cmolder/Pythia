"""
Utility functions for evaluating Condor experiments on Pythia.

Authors: Quang Duong and Carson Molder
"""
from collections import defaultdict
import os
import glob
import pandas as pd
import numpy as np

def read_file(path, cpu=0, cache_level='LLC'):
    expected_keys = ('ipc', 'total_miss', 'useful', 'useless', 'issued_prefetches', 'load_miss', 'rfo_miss', 'kilo_inst')
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if 'Finished CPU' in line:
                data['ipc'] = float(line.split()[9])
                data['kilo_inst'] = int(line.split()[4]) / 1000
            if f'Core_{cpu}_{cache_level}' not in line: # TODO : Implement multi-core support.
                continue
            line = line.strip()
            if 'load_miss' in line:
                data['load_miss'] = int(line.split()[1])
            elif 'RFO_miss' in line:
                data['rfo_miss'] = int(line.split()[1])
            elif 'total_miss' in line:
                data['total_miss'] = int(line.split()[1])
            elif 'prefetch_useful' in line:
                data['useful'] = int(line.split()[1])
            elif 'prefetch_useless' in line:
                data['useless'] = int(line.split()[1])
            elif 'prefetch_issued' in line:
                data['issued_prefetches'] = int(line.split()[1])

    if not all(key in data for key in expected_keys):
        return None

    return data
    

def get_statistics(path, baseline_path=None):
    full_trace = os.path.basename(path).split('-')[0]
    trace = full_trace.split('_')[0]
    simpoint = full_trace.split('_')[1] if len(full_trace.split('_')) >= 2 else None # TODO : Handle spec-17 formatting
    prefetcher = get_prefetcher_from_path(path)
    
    # Don't compare baseline to itself.
    if baseline_path and get_prefetcher_from_path(baseline_path) == prefetcher:
        return None 
    
    # Get statistics
    pf_data = read_file(path)
        
    #print('[get_statistics DEBUG]', trace, stats['simpoint'], stats['prefetcher'], pf_data)
    iss_prefetches, useful, useless, ipc, load_miss, rfo_miss, kilo_inst = (
        pf_data['issued_prefetches'], pf_data['useful'], pf_data['useless'], 
        pf_data['ipc'], pf_data['load_miss'], pf_data['rfo_miss'], pf_data['kilo_inst']
    )
    pf_total_miss = load_miss + rfo_miss + useful
    total_miss = pf_total_miss
    pf_mpki = (load_miss + rfo_miss) / kilo_inst
    
    if baseline_path:
        b_data = read_file(baseline_path)
        b_total_miss, b_ipc = b_data['total_miss'], b_data['ipc']
        b_load_miss = b_data['load_miss']
        b_rfo_miss = b_data['rfo_miss']
        b_mpki = b_total_miss / b_data['kilo_inst']
        assert b_data['kilo_inst'] == pf_data['kilo_inst'], 'Traces did not run for the same amount of instructions.'
        
    if useful + useless == 0:
        acc = 'N/A'
    else:
        acc = useful / (useful + useless) * 100
        
    if total_miss == 0 or baseline_path is None:
        cov = np.nan
        oldcov = np.nan
    else:
        cov = ((b_load_miss + b_rfo_miss) - (load_miss + rfo_miss)) / (b_load_miss + b_rfo_miss) * 100
        oldcov = str(useful / total_miss * 100)
        
    if baseline_path:
        mpki_reduction = (b_mpki - pf_mpki) / b_mpki * 100.
        ipc_improvement = (ipc - b_ipc) / b_ipc * 100.
    else:
        mpki_reduction = np.nan
        ipc_improvement = np.nan
        
    return {
        'full_trace': full_trace,
        'trace': trace,
        'simpoint': simpoint,
        'prefetcher': prefetcher,
        'accuracy': acc,
        'coverage': cov,
        'mpki': pf_mpki,
        'mpki_reduction': mpki_reduction,
        'ipc': ipc,
        'ipc_improvement': ipc_improvement,
        'baseline_prefetcher': get_prefetcher_from_path(baseline_path) if baseline_path else None,
        'path': path,
        'baseline_path': baseline_path
    }
    
    
def get_prefetcher_from_path(path):
    if 'multi' not in path:
        return 'no'
    else:
        return os.path.basename(path).split('-')[-1].rstrip('.txt')
    

def generate_csv(results_dir, output_file, baseline_prefetcher='no', dry_run=False):
    traces = defaultdict(dict)
    
    for path in glob.glob(os.path.join(results_dir, '*.txt')):
        full_trace = os.path.basename(path).split('-')[0]
        prefetcher = get_prefetcher_from_path(path)
          
        traces[full_trace][prefetcher] = path

            
    columns = ['full_trace', 'trace', 'simpoint', 'prefetcher', 'accuracy', 'coverage', 'mpki',
               'mpki_reduction', 'ipc', 'ipc_improvement', 'baseline_prefetcher', 'path', 'baseline_path']
    stats = pd.DataFrame(columns=columns)
    
    for tr in traces:
        assert baseline_prefetcher in traces[tr].keys(), f'Could not find baseline {baseline_prefetcher} run for trace {tr}'
        
        for pf in traces[tr]:
            row = get_statistics(traces[tr][pf], baseline_path=traces[tr][baseline_prefetcher])
            if row is None:
                continue
                
            assert all([k in columns for k in row.keys()])
            stats.loc[len(stats.index)] = row.values()
            
    if not dry_run:
        print(f'Saving statistics to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)
            
        
            
        

