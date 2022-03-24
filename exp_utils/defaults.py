import os

default_llc_repl_fn = 'ship'
default_results_dir = './out/results'
default_output_file = './out/stats.csv'
default_warmup_pct = 10
default_instrs = 50

default_llc_ways = 16
default_llc_sets = 2048

default_binary = 'bin/perceptron-no-no-{llc_pref_fn}-{llc_repl_fn}-{n_cores}core'
llc_sets_suffix = '-{n_sets}llc_sets'
llc_ways_suffx = '-{n_ways}llc_ways'

#prefetcher_names = ['no', 'bo', 'bingo', 'misb', 'pythia', 'spp', 'sms', 'triage']
#prefetcher_fns = ['no', 'bop', 'bingo', 'misb', 'scooby', 'spp_dev2', 'sms', 'triage'] # TODO - Double check in prefetcher/
prefetcher_names = ['no', 'multi']
prefetcher_fns = ['no', 'multi']
default_max_hybrid = 2

default_exp_dir = '/scratch/cluster/cmolder/prefetching_zoo/'
default_champsim_dir = os.getenv('PYTHIA_HOME')

default_prefetcher_candidates = [
    'no',
    'bingo',
    'bop',
    'scooby',
    'sisb',
    'spp_dev2',
]
    