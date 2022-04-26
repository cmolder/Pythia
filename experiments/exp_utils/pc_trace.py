import os
import pandas as pd
from tqdm import tqdm

metrics = ['num_useful', 'marginal_useful', 'accuracy']


def get_pc_trace_file(trace, metric, level='llc'):
    """Return name of a pc trace file.
    """
    return f'{trace}_{metric}_{level}_pc_trace.txt'
    


def _format_prefetcher(pref):
    
    pref = pref.replace('spp_dev2', 'sppdev2')
    pref = pref.replace('_', ',')
    pref = pref.replace('sppdev2', 'spp_dev2')
    return pref
    
def _format_degree(deg):
    
    deg = eval(deg)
    deg = ','.join([str(d) if d != None else 'na' for d in deg])
    return deg

def build_pc_traces(pc_stats_file, output_dir, metric, level='llc', dry_run=False):
    
    assert metric in metrics, f'Metric {metric} not in supported metrics {metrics}'   
    data = pd.read_csv(pc_stats_file)
    
    benchmarks = sorted(data.full_trace.unique().tolist())
    
    print('Building PC traces for...')
    print('    pc_stats_file:', pc_stats_file)
    print('    output_dir   :', output_dir)
    print('    metric       :', metric)
    print('    benchmarks   :', ', '.join(benchmarks))

    with tqdm(dynamic_ncols=True, unit='pc') as pbar:
        for benchmark in benchmarks:
            best_prefetcher = {}
            best_degree = {}
            bk_data = data[data.full_trace == benchmark]

            # Build dataset
            for pc in bk_data.pc.unique():
                pc_data = bk_data[bk_data.pc == pc]

                # Accuracy:
                # 1. Pick the prefetcher with the highest accuracy
                # 2. On a tie, pick the prefetcher with the highest number of useful prefetches.
                # 3. If still tied, pick one at random.
                if metric == 'accuracy':
                    best_pf = pc_data[pc_data.accuracy == pc_data.accuracy.max()]
                    best_pf = best_pf[best_pf.num_useful == best_pf.num_useful.max()]
                    best_pf = best_pf.sample(n = 1)

                # Num Useful:
                # 1. Pick the prefetcher with the largest number of useful prefetches
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
                    pc_data['marginal_useful'] = pc_data.num_useful - pc_data.num_useless
                    best_pf = pc_data[pc_data.marginal_useful == pc_data.marginal_useful.max()]
                    best_pf = pc_data[pc_data.num_useful == pc_data.num_useful.max()]
                    best_pf = best_pf.sample(n = 1)

                best_prefetcher[pc] = best_pf.pref.item()
                best_degree[pc] = best_pf.pref_degree.item()
                pbar.update(1)
            
            # Write PC traces
            # Lines are of the form {pc} {best_prefetcher} {best_degree}
            output_file = os.path.join(output_dir, get_pc_trace_file(benchmark, metric, level))
            if not dry_run:
                os.makedirs(output_dir, exist_ok=True)
                with open(output_file, 'w') as f:
                    for pc in best_prefetcher.keys():
                        print(f'{pc} {_format_prefetcher(best_prefetcher[pc])} {_format_degree(best_degree[pc])}', file=f)
        