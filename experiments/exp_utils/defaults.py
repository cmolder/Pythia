import os

binary_base = './bin/perceptron-no-no-{llc_pref_fn}-{llc_repl_fn}-{n_cores}core'
llc_sets_suffix = '-{llc_n_sets}llc_sets'
llc_ways_suffx = '-{llc_n_ways}llc_ways'

default_llc_repl_fn = 'ship'
default_llc_ways = 16
default_llc_sets = 2048

prefetcher_names = ['no', 'multi']
prefetcher_fns = ['no', 'multi']

default_results_dir = './out/'
default_config_file = './config/prefetcher_zoo.ini'
default_output_file = './out/stats.csv'
default_warmup_instructions = 10
default_sim_instructions = 50

# TODO: The below are legacy and need to be removed
# (should be defined in experiments/exp_config/*.yml)
default_max_hybrid = 2

default_prefetcher_candidates = [
    'no',       # No prefetcher
    'bingo',    # Bingo
    'bop',      # Best Offset
    'scooby',   # Pythia
    'sisb',     # Idealized ISB
    'sms',      # SMS
    'spp_dev2', # SPP
    'triage',   # Triage
]
    
default_condor_user = 'cmolder@cs.utexas.edu'
default_exp_dir = '/scratch/cluster/cmolder/prefetching_zoo/'
default_trace_dir = '/scratch/cluster/cmolder/traces/prefetching_zoo/champsim/'
default_champsim_dir = '/u/cmolder/GitHub/Pythia/'
default_conda_source = '/u/cmolder/miniconda3/etc/profile.d/conda.sh'
