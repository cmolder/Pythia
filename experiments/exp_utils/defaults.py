# Try not to import anything outside Python default libraries.
# Do not edit these to run programs, instead using the proper arguments, options, config files.

binary_base = './bin/{branch_pred}-{l1d_pref}-{l2c_pref}-{llc_pref}-{llc_repl}-{n_cores}core'
llc_sets_suffix = '-{llc_n_sets}llc_sets'
llc_ways_suffx = '-{llc_n_ways}llc_ways'

l1d_pref_fns = ['no', 'multi']
l2c_pref_fns = ['no', 'multi']
llc_pref_fns = ['no', 'multi', 'multi_pc_trace', 'from_file']
llc_repl_fns = ['ship', 'srrip', 'drrip', 'lru']

default_llc_ways = 16
default_llc_sets = 2048

default_results_dir = './out/'
default_knobs_file = './config/default_sweep.ini'
default_output_file = './out/stats.csv'
default_warmup_instructions = 10
default_sim_instructions = 50