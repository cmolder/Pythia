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
def read_file(path, cpu=0):
    expected_keys = ('ipc', 'total_miss', 'dram_bw_epochs', 
                     'L1D_useful', 'L2C_useful', 'LLC_useful',
                     'L1D_useless', 'L2C_useless', 'LLC_useless',
                     'L1D_issued_prefetches', 'L2C_issued_prefetches', 'LLC_issued_prefetches',
                     'L1D_load_miss', 'L2C_load_miss', 'LLC_load_miss', 
                     'L1D_rfo_miss', 'L2C_rfo_miss', 'LLC_rfo_miss',
                     'kilo_inst')
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
            if f'Core_{cpu}' not in line: # TODO : Implement multi-core support.
                continue
            line = line.strip()
            
            for level in ['L1D', 'L2C', 'LLC']:
                if f'{level}_load_miss' in line:
                    data[f'{level}_load_miss'] = int(line.split()[1])
                elif f'{level}_RFO_miss' in line:
                    data[f'{level}_rfo_miss'] = int(line.split()[1])
                elif f'{level}_total_miss' in line:
                    data['total_miss'] = int(line.split()[1])
                elif f'{level}_prefetch_useful' in line:
                    data[f'{level}_useful'] = int(line.split()[1])
                elif f'{level}_prefetch_useless' in line:
                    data[f'{level}_useless'] = int(line.split()[1])
                elif f'{level}_prefetch_issued' in line:
                    data[f'{level}_issued_prefetches'] = int(line.split()[1])

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
stats_columns = [
    'full_trace', 'trace', 'simpoint', 
    'L1D_pref', 'L1D_accuracy', 'L1D_coverage', 'L1D_mpki', 'L1D_mpki_reduction',
    'L2C_pref', 'L2C_pref_degree', 'L2C_accuracy', 'L2C_coverage', 'L2C_mpki', 'L2C_mpki_reduction',
    'LLC_pref', 'LLC_pref_degree', 'LLC_accuracy', 'LLC_coverage', 'LLC_mpki', 'LLC_mpki_reduction',
    'dram_bw_epochs', 'dram_bw_reduction', 
    'ipc', 'ipc_improvement',
    'pythia_level_threshold', 'pythia_high_conf_prefetches', 'pythia_low_conf_prefetches'
    'path', 'baseline_path'
]
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
    
    results = {k : np.nan for k in stats_columns}
    results['full_trace'] = full_trace
    results['trace'] = trace
    results['simpoint'] = simpoint
    results['pythia_level_threshold'] = pf_data['pythia_level_threshold']
    results['pythia_high_conf_prefetches'] = pf_data['pythia_high_conf_prefetches']
    results['pythia_low_conf_prefetches'] = pf_data['pythia_low_conf_prefetches']
    
    if baseline_path:
        b_data = read_file(baseline_path)
        b_ipc, b_dram_bw_epochs, b_kilo_inst = (
            b_data['ipc'], b_data['dram_bw_epochs'], b_data['kilo_inst']
        )
    
    ipc, dram_bw_epochs, kilo_inst = (
        pf_data['ipc'], pf_data['dram_bw_epochs'], pf_data['kilo_inst']
    )


    
    for level in ['L1D', 'L2C', 'LLC']:
        iss_prefetches, useful, useless, load_miss, rfo_miss = (
            pf_data[f'{level}_issued_prefetches'],
            pf_data[f'{level}_useful'],  pf_data[f'{level}_useless'],  
            pf_data[f'{level}_load_miss'], pf_data[f'{level}_rfo_miss']
        )
        
        total_miss = load_miss + rfo_miss + useful
        pf_mpki = (load_miss + rfo_miss) / kilo_inst
        
        if baseline_path:
            b_load_miss, b_rfo_miss, b_useful = (
               b_data[f'{level}_load_miss'], b_data[f'{level}_rfo_miss'], b_data[f'{level}_useful']
            )
            b_total_miss = load_miss + rfo_miss + useful # = b_data[f'{level}_total_miss']
            b_mpki = b_total_miss / b_data['kilo_inst']
            assert np.isclose(b_data['kilo_inst'], pf_data['kilo_inst']), f'Traces {os.path.basename(path)}, {os.path.basename(baseline_path)} did not run for the same amount of instructions. ({b_data["kilo_inst"]}K vs {pf_data["kilo_inst"]}K)'

        if level == 'L1D':
            results[f'{level}_pref'] = l1d_pref
        elif level == 'L2C':
            results[f'{level}_pref'] = l2c_pref
            results[f'{level}_pref_degree'] = l2c_pref_degree
        else:
            results[f'{level}_pref'] = llc_pref
            results[f'{level}_pref_degree'] = llc_pref_degree
        
        results[f'{level}_accuracy'] = 100.0 if (useful + useless == 0) else useful / (useful + useless) * 100.
        results[f'{level}_coverage'] = np.nan if (total_miss == 0 or baseline_path is None) else ((b_load_miss + b_rfo_miss) - (load_miss + rfo_miss)) / (b_load_miss + b_rfo_miss) * 100
        results[f'{level}_mpki'] = pf_mpki
        results[f'{level}_mpki_reduction'] = np.nan if (baseline_path is None) else (b_mpki - pf_mpki) / b_mpki * 100.
                
        
    results['dram_bw_epochs'] = dram_bw_epochs
    results['dram_bw_reduction'] = np.nan if (baseline_path is None) else (b_dram_bw_epochs - dram_bw_epochs) / b_dram_bw_epochs * 100.
    results['ipc'] = ipc
    results['ipc_improvement'] = np.nan if (baseline_path is None) else (ipc - b_ipc) / b_ipc * 100.  
    results['path'] = path
    results['baseline_path'] = baseline_path 
    return results

pc_columns = [
    'pc', 'full_trace', 'trace', 'simpoint', 
    'pref', 'pref_degree', 'num_useful', 'num_useless', 'accuracy',
]
def get_pc_statistics(path):
    """Get per-PC statistics from a Champsim pc_pref_stats file.
    """
    full_trace = os.path.basename(path).split('-')[0]
    trace, simpoint = get_trace_from_path(path)
    l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
    l2c_pref_degree, llc_pref_degree = get_prefetcher_degs_from_path(path)
    
    if path.endswith('l1d.txt'):
        pref = l1d_pref
        pref_degree = None
    elif path.endswith('l2d.txt'):
        pref = l2d_pref,
        pref_degree = l2d_pref_degree
    else:
        pref = llc_pref
        pref_degree = llc_pref_degree
    
    # Get statistics
    pc_data = read_pc_file(path)
    pc_out = []
    
    for pc in pc_data.keys(): 
        row = {k : np.nan for k in pc_columns}
        row['pc'] = pc
        row['full_trace'] = full_trace
        row['trace'] = trace
        row['simpoint'] = simpoint
        row['pref'] = pref
        row['pref_degree'] = pref_degree
        row['num_useful'] = pc_data[pc]['useful']
        row['num_useless'] =  pc_data[pc]['useless']
        row['accuracy'] = 100.0 if (pc_data[pc]['useful'] + pc_data[pc]['useless'] == 0) else (pc_data[pc]['useful'] / (pc_data[pc]['useful'] + pc_data[pc]['useless']))
        pc_out.append(row)
        
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
def generate_csv(results_dir, output_file, dry_run=False):
    """Generate cumulative statistics for each run.
    """
    traces = defaultdict(lambda : defaultdict(dict))
    n_paths = len(glob.glob(os.path.join(results_dir, '*.txt')))
    
    # Build trace paths
    for path in glob.glob(os.path.join(results_dir, '*.txt')):
        full_trace = os.path.basename(path).split('-')[0]
        l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
        l2c_pref_deg, llc_pref_deg = get_prefetcher_degs_from_path(path)
        pyt_level_th = get_pythia_level_threshold(path)
        traces[full_trace][(l1d_pref, l2c_pref, llc_pref)][(l2c_pref_deg, llc_pref_deg, pyt_level_th)] = path

        
    # Build statistics table
    stats = []
    with tqdm(total=n_paths, dynamic_ncols=True, unit='trace') as pbar:
        for tr in traces:
            assert ('no', 'no', 'no') in traces[tr].keys(), f'Could not find baseline ("no", "no", "no") run for trace {tr}'
            for pf in traces[tr]:
                for d in traces[tr][pf]:
                    pbar.update(1)

                    row = get_statistics(traces[tr][pf][d], baseline_path=traces[tr][('no', 'no', 'no')][((None,), (None,), None)])         
                    if row is None: # Filter missing rows
                        continue
                        
                    assert all([k in stats_columns for k in row.keys()]), f'Columns missing for row {tr}, {pf}, {d}'
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
        l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
        l2c_pref_degrees, llc_pref_degrees = get_prefetcher_degs_from_path(path)
        traces[full_trace][(l1d_pref, l2c_pref, llc_pref)][(l2c_pref_degrees, llc_pref_degrees)] = path

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
       
                
def generate_pc_csv(results_dir, output_file, level='llc', dry_run=False):
    """Generate statistics on each PC for each prefetcher on each run.
    """
    traces = defaultdict(lambda : defaultdict(dict))
    paths = glob.glob(os.path.join(results_dir, 'pc_pref_stats', f'*_{level}.txt'))
    n_paths = len(paths)
    
    # Build trace paths
    for path in paths:
        full_trace = os.path.basename(path).split('-')[0]
        l1d_pref, l2c_pref, llc_pref = get_prefetcher_from_path(path)
        l2c_pref_degrees, llc_pref_degrees = get_prefetcher_degs_from_path(path)
        
        traces[full_trace][(l1d_pref, l2c_pref, llc_pref)][(l2c_pref_degrees, llc_pref_degrees)] = path
       
    # Build statistics table
<<<<<<< HEAD
    # If you edit the columns, preserve the ordering between the dict in get_statistics and this list.
    columns = ['pc', 'full_trace', 'trace', 'simpoint', 
               'l1d_pref', 'l2c_pref', 'llc_pref',
               'l2c_pref_degree', 'llc_pref_degree', 
               'num_useful', 'num_useless', 'accuracy']
=======
>>>>>>> master
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