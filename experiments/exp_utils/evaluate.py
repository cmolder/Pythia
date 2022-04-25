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



"""
File I/O helpers
"""
def read_file(path, cpu=0, cache_level='LLC'):
    expected_keys = ('ipc', 'total_miss', 'dram_bw_epochs', 'useful', 'useless', 'issued_prefetches', 'load_miss', 'rfo_miss', 'kilo_inst') 

    data = {}
    
    # Optional data (not checked in expected_keys)
    pythia_dyn_level = False
    data['pythia_level_threshold'] = None
    data['pythia_low_conf_prefetches'] = None
    data['pythia_high_conf_prefetches'] = None
    
    with open(path, 'r') as f:
        for line in f:
            if 'Finished CPU' in line:
                data['ipc'] = float(line.split()[9])
                data['kilo_inst'] = int(line.split()[4]) / 1000
            if 'DRAM_bw_pochs' in line:
                data['dram_bw_epochs'] = int(line.split()[1])
            if 'scooby_enable_dyn_level 1' in line:
                pythia_dyn_level = True
            if 'scooby_dyn_level_threshold' in line and pythia_dyn_level:
                data['pythia_level_threshold'] = float(line.split()[1])
            if 'scooby_low_conf_pref' in line:
                data['pythia_low_conf_prefetches'] = int(line.split()[1])
            if 'scooby_high_conf_pref' in line:
                data['pythia_high_conf_prefetches'] = int(line.split()[1])
                
            
            # Per-core, cache-level statistics
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


def read_pc_file(path):
    """Read a PC prefetcher statistics file and return the data.
    """
    pc_stats = defaultdict(dict)

    with open(path, 'r') as f:
        for line in f:
            pc = line.split()[0]
            pc_stats[pc]['useful'] =  int(line.split()[1])
            pc_stats[pc]['useless'] = int(line.split()[2])
            
    return pc_stats

    
"""
Statistics helpers
"""
def get_statistics(path, baseline_path=None):
    """Get cumulative statistics from a ChampSim output / results file.
    """
    full_trace = os.path.basename(path).split('-')[0]
    trace, simpoint = get_trace_from_path(path)
    l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
    l2c_pref_degree, llc_pref_degree = get_prefetcher_degs_from_path(path)
    
    # Don't compare baseline to itself.
    if baseline_path and get_prefetcher_from_path(baseline_path) == (l1d_pref, l2c_pref, llc_pref):
        return None 
    
    # Get statistics
    pf_data = read_file(path)
    if pf_data == None:
        print(f'Warning: Missing data for {path}')
        return None
        
        
    iss_prefetches, useful, useless, ipc, load_miss, rfo_miss, dram_bw_epochs, kilo_inst = (
        pf_data['issued_prefetches'], pf_data['useful'], pf_data['useless'], 
        pf_data['ipc'], pf_data['load_miss'], pf_data['rfo_miss'], 
        pf_data['dram_bw_epochs'], pf_data['kilo_inst'],
    )
    pythia_level_threshold, pythia_high_conf_prefetches, pythia_low_conf_prefetches = (
        pf_data['pythia_level_threshold'],  pf_data['pythia_high_conf_prefetches'],
        pf_data['pythia_low_conf_prefetches']
    )
    
    
    pf_total_miss = load_miss + rfo_miss + useful
    total_miss = pf_total_miss
    pf_mpki = (load_miss + rfo_miss) / kilo_inst
    
    if baseline_path:
        b_data = read_file(baseline_path)
        b_total_miss, b_ipc = b_data['total_miss'], b_data['ipc']
        b_dram_bw_epochs = b_data['dram_bw_epochs']
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
        dram_bw_reduction = (b_dram_bw_epochs - dram_bw_epochs) / b_dram_bw_epochs * 100.
        ipc_improvement = (ipc - b_ipc) / b_ipc * 100.
    else:
        mpki_reduction = np.nan
        dram_bw_reduction = np.nan
        ipc_improvement = np.nan
        
    return {
        'full_trace': full_trace,
        'trace': trace,
        'simpoint': simpoint,
        'l1d_pref': l1d_pref,
        'l2c_pref': l2c_pref,
        'llc_pref': llc_pref,
        'l2c_pref_degree': l2c_pref_degree,
        'llc_pref_degree': llc_pref_degree,
        'pythia_level_threshold': pythia_level_threshold,
        'accuracy': acc,
        'coverage': cov,
        'pythia_low_conf_prefetches': pythia_low_conf_prefetches,
        'pythia_high_conf_prefetches': pythia_high_conf_prefetches,
        'mpki': pf_mpki,
        'mpki_reduction': mpki_reduction,
        'dram_bw_epochs': dram_bw_epochs,
        'dram_bw_reduction': dram_bw_reduction,
        'ipc': ipc,
        'ipc_improvement': ipc_improvement,
        'baseline_prefetcher': get_prefetcher_from_path(baseline_path) if baseline_path else None,
        'path': path,
        'baseline_path': baseline_path
    }


def get_pc_statistics(path):
    """Get per-PC statistics from a Champsim pc_pref_stats file.
    """
    full_trace = os.path.basename(path).split('-')[0]
    trace, simpoint = get_trace_from_path(path)
    l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
    l2c_pref_degree, llc_pref_degree = get_prefetcher_degs_from_path(path)
    
    # Get statistics
    pc_data = read_pc_file(path)
    pc_out = []
    
    for pc in pc_data.keys():
        useful, useless = pc_data[pc]['useful'], pc_data[pc]['useless']
        
        if useful + useless == 0:
            accuracy = 100.0 # 'N/A'
        else:
            accuracy = useful / (useful + useless)
            
        # TODO: Coverage
        
        pc_out.append({
            'pc': pc,
            'full_trace': full_trace,
            'trace': trace,
            'simpoint': simpoint,
            'l1d_pref': l1d_pref,
            'l2c_pref': l2c_pref,
            'llc_pref': llc_pref,
            'l2c_pref_degree': l2c_pref_degree,
            'llc_pref_degree': llc_pref_degree,
            'num_useful': useful,
            'num_useless': useless,
            'accuracy': accuracy
        })
        
    return pc_out
    
    
    
    

"""
Path helper functions
"""
def get_full_trace_from_path(path):
    """Get the full trace name, including simpoint.
    """
    trace = os.path.basename(path).split('-')[0]
    return trace


def get_trace_from_path(path):
    """Get the trace name and simpoint name.
    
    For multicore traces (e.g. CloudSuite),
    will return trace name as name + core.
    
    (TODO: Handle SPEC '17 formatting)
    """
    trace = os.path.basename(path).split('-')[0]
    tokens = trace.split('_')
    if len(tokens) == 1: # Gap        : e.g. bfs
        return tokens[0], None
    if len(tokens) == 2: # SPEC '06   : e.g. astar_313B
        return tokens[0], tokens[1]
    if len(tokens) == 3: # Cloudsuite : e.g. cassandra_phase0_core0
        return tokens[0] + '_' + tokens[2], tokens[1] # Name + core, simpoint
    
    
def get_prefetcher_from_path(path):
    """Get the prefetcher(s) name(s).
    """
    l1p, l2p, llp = os.path.basename(path).split('-')[2:5]
    
    if llp not in ['no', 'multi_pc_trace']:
        llp, _ = llp.replace('spp_dev2', 'sppdev2').split('_')
        llp  = llp.replace(',','_').replace('sppdev2', 'spp_dev2')
    if l2p not in ['no']:
        l2p, _ = l2p.replace('spp_dev2', 'sppdev2').split('_')
        l2p  = l2p.replace(',','_').replace('sppdev2', 'spp_dev2')
    
    
    return l1p, l2p, llp
    
    
def get_prefetcher_degs_from_path(path):
    """Get the prefetcher(s) degree(s).
    """
    l1p, l2p, llp = os.path.basename(path).split('-')[2:5]
    
    l2pd, llpd = (None,), (None,)
    
    if l2p not in ['no']:
        _, l2pd = l2p.replace('spp_dev2', 'sppdev2').split('_')
        l2pd = tuple((None if d == 'na' else int(d)) for d in l2pd.split(','))
    if llp not in ['no', 'multi_pc_trace']:
        _, llpd = llp.replace('spp_dev2', 'sppdev2').split('_')
        llpd = tuple((None if d == 'na' else int(d)) for d in llpd.split(','))
    
    return l2pd, llpd

def get_pythia_level_threshold(path):
    """Get the level threshold for Pythia,
    if is is being used.
    """
    if not any(p == 'scooby' for p in get_prefetcher_from_path(path)):
        return None
    
    if 'threshold' in path:
        path_ = path[path.index('threshold'):]
        path_ = path_.replace('threshold_', '').replace('.txt', '')
        return float(path_)
    
    return None

"""
CSV file creators
"""
def generate_csv(results_dir, output_file, 
                 best_degree_csv_file=None, dry_run=False):
    """Generate cumulative statistics for each run.
    """
    traces = defaultdict(lambda : defaultdict(dict))
    n_paths = len(glob.glob(os.path.join(results_dir, '*.txt')))
    
    # If we have a best degree CSV file, we will filter out instances of the prefetchers
    # found in the file on the ones with the best degree in each trace.
    if best_degree_csv_file:
        best_deg_df = pd.read_csv(best_degree_csv_file)
        best_deg_df.index = best_deg_df.Trace
        #print(best_deg_df)
    
    # Build trace paths
    for path in glob.glob(os.path.join(results_dir, '*.txt')):
        full_trace = os.path.basename(path).split('-')[0]
        l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
        l2c_pref_deg, llc_pref_deg = get_prefetcher_degs_from_path(path)
        pyt_level_th = get_pythia_level_threshold(path)
        traces[full_trace][(l1d_pref, l2c_pref, llc_pref)][(l2c_pref_deg, llc_pref_deg, pyt_level_th)] = path
        
        
    # Build statistics table
    # If you edit the columns, preserve the ordering between the dict in get_statistics and this list.
    columns = ['full_trace', 'trace', 'simpoint', 
               'l1d_pref', 'l2c_pref', 'llc_pref',
               'l2c_pref_degree', 'llc_pref_degree', 
               'pythia_level_threshold', 
               'accuracy', 'coverage', 
               'pythia_low_conf_prefetches', 'pythia_high_conf_prefetches',
               'mpki', 'mpki_reduction', 
               'dram_bw_epochs', 'dram_bw_reduction', 
               'ipc', 'ipc_improvement',
               'baseline_prefetcher', 'path', 'baseline_path']
    stats = []
    with tqdm(total=n_paths, dynamic_ncols=True, unit='trace') as pbar:
        for tr in traces:
            assert ('no', 'no', 'no') in traces[tr].keys(), f'Could not find baseline "no" run for trace {tr}'
            for pf in traces[tr]:
                for d in traces[tr][pf]:
                    pbar.update(1)
                    
                    # Filter non-best degree prefetchers (if provided).
                    if best_degree_csv_file and pf in best_deg_df.columns and eval(best_deg_df[pf].loc[tr]) != d:
                        #print('[DEBUG] Skipping', pf, d, 'best is', best_deg_df[pf].loc[tr])
                        continue
                        
                    row = get_statistics(traces[tr][pf][d], baseline_path=traces[tr][('no', 'no', 'no')][((None,), (None,), None)])         

                    # Filter missing rows
                    if row is None:
                        continue
                        
                    assert all([k in columns for k in row.keys()]), f'Columns missing for row {tr}, {pf}, {d}'
                    stats.append(row)
                    
    # Build and save statistics table
    stats = pd.DataFrame(stats)
    if not dry_run:
        print(f'Saving statistics table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)
            
            
def generate_best_degree_csv(results_dir, output_file, 
                             metric='ipc', dry_run=False):
    """Generate the best degree for each prefetcher on each run.
    """
    traces = defaultdict(lambda : defaultdict(dict))
    prefetchers = set()
    best_degree = defaultdict(dict)
    paths = glob.glob(os.path.join(results_dir, '*.txt'))
    n_paths = len(paths)

    # Build trace paths
    for path in paths:
        full_trace = os.path.basename(path).split('-')[0]
        prefetcher = get_prefetcher_from_path(path)
        degrees = get_prefetcher_degs_from_path(path)
        traces[full_trace][prefetcher][degrees] = path

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
        
        
def generate_pc_csv(results_dir, output_file, level='llc', 
                    dry_run=False, best_degree_csv_file=None):
    """Generate statistics on each PC for each prefetcher on each run.
    """
    traces = defaultdict(lambda : defaultdict(dict))
    paths = glob.glob(os.path.join(results_dir, 'pc_pref_stats', f'*_{level}.txt'))
    n_paths = len(paths)
    
    # If we have a best degree CSV file, we will filter out instances of the prefetchers
    # found in the file on the ones with the best degree in each trace.
    if best_degree_csv_file:
        best_deg_df = pd.read_csv(best_degree_csv_file)
        best_deg_df.index = best_deg_df.Trace
    
    # Build trace paths
    for path in paths:
        full_trace = os.path.basename(path).split('-')[0]
        prefetcher = get_prefetcher_from_path(path)
        degrees = get_prefetcher_degs_from_path(path)
        
        traces[full_trace][prefetcher][degrees] = path
       
    # Build statistics table
    # If you edit the columns, preserve the ordering between the dict in get_statistics and this list.
    columns = ['pc', 'full_trace', 'trace', 'simpoint', 
               'l1d_pref', 'l2c_pref', 'llc_pref',
               'l2c_pref_degree', 'llc_pref_degree', 
               'num_useful', 'num_useless', 'accuracy']
    stats = []
    with tqdm(total=n_paths, dynamic_ncols=True, unit='trace') as pbar:
        for tr in traces.keys():
            for pf in traces[tr].keys():
                if pf == 'no':
                    pbar.update(1)
                    continue
                
                for d in traces[tr][pf].keys():
                    pbar.update(1)
                    # Filter non-best degree prefetchers (if provided).
                    if best_degree_csv_file and pf in best_deg_df.columns and eval(best_deg_df[pf].loc[tr]) != d:
                        continue
                        
                    rows = get_pc_statistics(traces[tr][pf][d])
                    stats.extend(rows)

    # Save statistics table
    stats = pd.DataFrame(stats)
    if not dry_run:
        print(f'Saving per-PC {level} prefetch statistics table to {output_file}...')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        stats.to_csv(output_file, index=False)