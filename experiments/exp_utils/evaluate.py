"""
Utility functions for evaluating Condor experiments on Pythia.

Authors: Quang Duong and Carson Molder
"""
from collections import defaultdict
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    degree = get_prefetcher_degs_from_path(path)
    
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
        assert np.isclose(b_data['kilo_inst'], pf_data['kilo_inst']), f'Traces {os.path.basename(path)}, {os.path.basename(baseline_path)} did not run for the same amount of instructions. ({b_data["kilo_inst"]}K vs {pf_data["kilo_inst"]}K)'
        
    if useful + useless == 0:
        acc = 100.0 #'N/A'
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
        'degree': degree,
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
    l1p, l2p, llp = os.path.basename(path).split('-')[2:5]
    
    if llp == 'no':
        return llp
    
    llp, _ = llp.replace('spp_dev2', 'sppdev2').split('_')
    llp  = llp.replace(',','_').replace('sppdev2', 'spp_dev2')
    return llp

    
def get_prefetcher_degs_from_path(path):
    l1p, l2p, llp = os.path.basename(path).split('-')[2:5]
    
    if llp == 'no':
        return (None,)
    
    _, llpd = llp.replace('spp_dev2', 'sppdev2').split('_')
    llpd = tuple((None if d == 'na' else int(d)) for d in llpd.split(','))
    
    return llpd
    

def generate_csv(results_dir, output_file, 
                 best_degree_csv_file=None, dry_run=False):
    traces = defaultdict(lambda : defaultdict(dict))
    n_paths = len(glob.glob(os.path.join(results_dir, '*.txt')))
    
    for path in glob.glob(os.path.join(results_dir, '*.txt')):
        full_trace = os.path.basename(path).split('-')[0]
        prefetcher = get_prefetcher_from_path(path)
        degrees = get_prefetcher_degs_from_path(path)
        traces[full_trace][prefetcher][degrees] = path
        
    # If we have a best degree CSV file, we will filter out instances of the prefetchers
    # found in the file on the ones with the best degree in each trace.
    if best_degree_csv_file:
        best_deg_df = pd.read_csv(best_degree_csv_file)
        best_deg_df.index = best_deg_df.Trace
        #print(best_deg_df)

            
    # Build statistics table
    columns = ['full_trace', 'trace', 'simpoint', 'prefetcher', 'degree', 'accuracy', 'coverage', 'mpki',
               'mpki_reduction', 'ipc', 'ipc_improvement', 'baseline_prefetcher', 'path', 'baseline_path']
    stats = pd.DataFrame(columns=columns)
    
    with tqdm(total=n_paths, dynamic_ncols=True) as pbar:
        for tr in traces:
            assert 'no' in traces[tr].keys(), f'Could not find baseline "no" run for trace {tr}'
            for pf in traces[tr]:
                for d in traces[tr][pf]:
                    #print('[DEBUG]', tr, pf, d)
                    row = get_statistics(traces[tr][pf][d], baseline_path=traces[tr]['no'][(None,)])
                    pbar.update(1)
                    
                    # Filter missing rows
                    if row is None:
                        continue
                    # Filter non-best degree prefetchers (if provided).
                    if best_degree_csv_file and pf in best_deg_df.columns and eval(best_deg_df[pf].loc[tr]) != d:
                        #print('[DEBUG] Skipping', pf, d, 'best is', best_deg_df[pf].loc[tr])
                        continue

                    assert all([k in columns for k in row.keys()]), f'Columns missing for row {tr}, {pf}, {d}'
                    stats.loc[len(stats.index)] = row.values()
                
    # Save statistics table
    if not dry_run:
        print(f'Saving statistics table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)
            

def generate_best_degree_csv(results_dir, output_file, 
                             metric='ipc', dry_run=False):
    traces = defaultdict(lambda : defaultdict(dict))
    prefetchers = set()
    best_degree = defaultdict(dict)
    n_paths = len(glob.glob(os.path.join(results_dir, '*.txt')))
    
    for path in glob.glob(os.path.join(results_dir, '*.txt')):
        full_trace = os.path.basename(path).split('-')[0]
        prefetcher = get_prefetcher_from_path(path)
        degrees = get_prefetcher_degs_from_path(path)
 
        traces[full_trace][prefetcher][degrees] = path

    # Build best degree dictionary
    # - Compute the best_degree for each prefetcher on each trace.
    with tqdm(total=n_paths, dynamic_ncols=True) as pbar:
        for tr in traces.keys():
            for pf in traces[tr].keys():
                if pf == 'no':
                    pbar.update(1)
                    continue
                prefetchers.add(pf)
                    
                scores = defaultdict(lambda : float('-inf'))
                for d in traces[tr][pf].keys():
                    pbar.update(1)    
                    row = get_statistics(traces[tr][pf][d], baseline_path=traces[tr]['no'][(None,)])
                    if row is None:
                        continue
                    scores[d] = row[metric]

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