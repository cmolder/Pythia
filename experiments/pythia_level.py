#!/usr/bin/env python3
"""
Set up and evaluate sweeps for the level-aware
Pythia prefetcher experiments.

Authors: Quang Duong and Carson Molder

TODO: Refactor to pythia.py
"""

import argparse
import sys
from exp_utils import condor, config, evaluate

# Defaults (TODO: Move to yml or launch args)
default_eval_csv = './out/pythia_level.csv'

help_str = {
'help':
'''usage: {prog} command [<args>]

Available commands:
    condor Set up Pythia Level sweep on Condor
    eval   Parse and compute metrics on sweep results
    help   Display this help message. Command-specific help messages
           can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),
'condor':
'''usage: {prog} condor <config-file> [-v / --verbose] [-d / --dry-run]

Description:
    {prog} condor <config-file>
        Sets up a Pythia Level sweep for use on Condor. <config-file> is 
        a path to a .yml file with the config 
        (example: experiments/exp_utils/pythia_level.yml)
        
Options:
    -v / --verbose
        If passed, prints extra details about the experiment setup.
        
    -d / --dry-run
        If passed, builds the experiment but writes nothing to 
        <experiment-dir>.∂
'''.format(prog=sys.argv[0], ),
'eval':
'''usage: {prog} eval <results-dir> [--output-file <out-file>] 
                                    [--norm-baseline <baseline>]

Description:
    {prog} eval <results-dir>
        Runs the evaluation procedure on the ChampSim result files found 
        in <results-dir> and outputs a CSV at the specified output path. 
        Determines the best degree for each prefetcher combination.

Options:
    -w / --weight-file <weights-file>
        Specifies SimPoint weights for the traces, and caluclates
        a weighted SimPoint average for the metrics, adding it
        to the ouput under a "weighted" SimPoint.

    -o / --output-file <output-file>
        Specifies what file path to save the stats CSV data to. 
        Default: `{default_output_file}`
        
    --dry-run
        If passed, builds the spreadsheet but writes nothing to 
        <output-file>.

Note:
    To get stats comparing performance to a no-prefetcher baseline, it 
    is necessary to have run the base ChampSim binary on the same 
    execution trace.

    Without the base data, relative performance data comparing MPKI and 
    IPC will not be available and the coverage statistic will only be 
    approximate.
'''.format(
        prog=sys.argv[0],
        default_output_file=default_eval_csv,
    ),
}
"""
Condor
"""


def condor_command():
    """Condor command
    """
    if len(sys.argv) < 3:
        print(help_str['condor'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('config_file', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    cfg = config.read_config(args.config_file)

    print('Setting up Condor Prefetcher Zoo experiment:')
    print('    ChampSim:')
    print('        # sim inst         :', cfg.champsim.sim_instructions, 'M')
    print('        # warmup inst      :', cfg.champsim.warmup_instructions, 'M')
    print('        track pc stats?    :', cfg.champsim.track_pc_pref)
    print('        track addr stats?  :', cfg.champsim.track_addr_pref)
    print('        seeds              :', cfg.champsim.seeds)
    print('    Directories:')
    print('        ChampSim           :', cfg.paths.champsim_dir)
    print('        Experiment         :', cfg.paths.exp_dir)
    print('        Traces             :', cfg.paths.trace_dir)
    print('    L1D:')
    print('        Pref. candidates   :', ', '.join(cfg.l1d.pref_candidates))
    print('        Max hybrid         :', cfg.l1d.max_hybrid)
    print('    L2C:')
    print('        Pref. candidates   :', ', '.join(cfg.l2c.pref_candidates))
    print('        Max hybrid         :', cfg.l2c.max_hybrid)
    print('    LLC:')
    print('        Sets               :', cfg.llc.sets)
    print('        Pref. candidates   :', ', '.join(cfg.llc.pref_candidates))
    print('        Replacement        :', cfg.llc.repl)
    print('        Max hybrid         :', cfg.llc.max_hybrid)
    print('    Pythia:')
    print('        Learning:')
    print('            Feature set(s) :', cfg.pythia.features)
    print('            Pooling        :', cfg.pythia.pooling)
    print('            Alpha          :', cfg.pythia.alpha)
    print('            Gamma          :', cfg.pythia.gamma)
    print('            Epsilon        :', cfg.pythia.epsilon)
    print('            Policy         :', cfg.pythia.policy)
    print('            Learning type  :', cfg.pythia.learning_type)
    print('        Level-awareness:')
    print('            Level threshs  :', cfg.pythia.dyn_level_threshold,
          '(also running without dynamic level)')
    print('            Separate PT    :', cfg.pythia.separate_lowconf_pt, 
          '(for low-confidence prefetches)')
    print('        Prefetch table:')
    print('            Size           :', cfg.pythia.pt_size)
    print('            Low conf. size :', cfg.pythia.lowconf_pt_size, 
          '(if separate PT enabled)')

    condor.build_pythia_sweep(cfg,
                              dry_run=args.dry_run,
                              verbose=args.verbose)


def eval_command():
    """Eval command
    
    Return statistics on each run of Pythia.
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('-w',
                        '--weight-file',
                        type=str,
                        default=None)
    parser.add_argument('-o',
                        '--output-file',
                        type=str,
                        default=default_eval_csv)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])

    evaluate.generate_run_csv(args.results_dir,
                              args.output_file,
                              weights_file=args.weight_file,
                              dry_run=args.dry_run)


def help_command():
    """Help command
    """
    # If one of the available help strings, print and exit successfully
    if len(sys.argv) > 2 and sys.argv[2] in help_str:
        print(help_str[sys.argv[2]])
        exit()
    # Otherwise, invalid subcommand, so print main help string and exit
    else:
        print(help_str['help'])
        exit(-1)


def main():
    commands = {
        'condor': condor_command,
        'eval': eval_command,
        'help': help_command,
    }
    # If no subcommand specified or invalid subcommand, print 
    # main help string and exit
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(help_str['help'])
        exit(-1)

    # Run specified subcommand
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
