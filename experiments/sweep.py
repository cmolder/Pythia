#!/usr/bin/env python3

"""
Set up and evaluate sweeps for ChampSim experiments on Condor.

Authors: Quang Duong and Carson Molder
"""

import argparse
import os
import sys

from exp_utils import condor, config, evaluate, pc_trace

# Defaults
# TODO: Move to yml or launch args
default_eval_csv = './out/sweep/sweep.csv'
default_pc_trace_metric = 'num_useful'
default_best_degree_metric = 'ipc'

help_str = {
    'help': '''usage: {prog} command [<args>]

Available commands:
    condor            Set up sweep on Condor
    
    eval              Parse and compute metrics on sweep results

    best_degree       Find the best degree for each prefetcher on each
                      trace, when comparing prefetch degrees
    
    online_pc_trace   Parse a per-PC statistics file and generate PC 
                      traces of the best prefetchers for each PC on each 
                      SimPoint, for use in online evaluation via 
                      multi_pc_trace
                      
    offline_pc_trace  Parse a per-PC statistics file and generate 
                      prefetch traces using the best prefetchers for 
                      each PC on each SimPoint, for use in offline 
                      evaluation via from_file
                      
    help              Display this help message. Command-specific help 
                      messages can be displayed with 
                      `{prog} help command`
'''.format(prog=sys.argv[0]),

    'condor': '''usage: {prog} condor <config-file> [-v / --verbose] 
                                                [-d / --dry-run]

Description:
    {prog} condor <config-file>
        Sets up a sweep for use on Condor. <config-file> 
        is a path to a .yml file with the config 
        (example: experiments/exp_utils/zoo.yml)
        
Options:
    -v / --verbose
        If passed, prints extra details about the experiment setup.
        
    -d / --dry-run
        If passed, builds the experiment but writes nothing to 
        <experiment-dir>.
'''.format(
        prog=sys.argv[0],
    ),

    'eval': '''usage: {prog} eval <results-dir> 
                              [--output-file <output-file>] 
                              [--norm-baseline <baseline>]

Description:
    {prog} eval <results-dir>
        Runs the evaluation procedure on the ChampSim result files found 
        in <results-dir> (i.e. champsim_results/) and outputs a CSV at 
        the specified output path.

Options:
    -w / --weight-file <weights-file>
        Specifies SimPoint weights for the traces, and caluclates
        a weighted SimPoint average for the metrics, adding it
        to the ouput under a "weighted" SimPoint.

    -o / --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This 
        defaults to `{default_eval_csv}`.
              
    --pc
        If provided, will compute per-PC prefetch stats on the LLC, 
        using results in <results-dir>/pc_pref_stats/
        
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
        default_eval_csv=default_eval_csv
    ),
    'best_degree': '''usage: {prog} eval <results-dir> 
                                     [--output-file <output-file>] 
                                     [--norm-baseline <baseline>]
Description:
    {prog} eval <results-dir>
        Runs the evaluation procedure on the ChampSim result files found 
        in <results-dir> and outputs a CSV at the specified output path. 
        Determines the best degree for each prefetcher combination.

Options:
    -o / --output-file <output-file>
        Specifies what file path to save the stats CSV data to. 
        Default: `{default_output_file}`
        
    -m / --metric <metric>
        Specifies the metric to compare prefetching degrees to.
        Default: `{default_metric}`
        
    --dry-run
        If passed, builds the spreadsheet but writes nothing to 
        <output-file>.
Note:
    To get stats comparing performance to a no-prefetcher baseline,
    it is necessary to have run the base ChampSim binary on the same
    execution trace. Without the baseline data, relative performance
    data comparing MPKI and IPC will not be available and coverage 
    will be approximate.

'''.format(
        prog=sys.argv[0],
        default_output_file=default_eval_csv,
        default_metric=default_best_degree_metric,
    ),
    'online_pc_trace': '''usage: {prog} online_pc_trace <pc-stats-file> 
                                    <output-dir> 
                                    [-m / --metric <metric>]

Description:
    {prog} online_pc_trace <pc-stats-file> <output-dir>
        Parses a PC stats file, and for each PC in each trace, 
        determines the best prefetcher under <metric>. These online 
        traces are saved to the output dir, for use in the 
        multi_pc_trace prefetcher.
        
        <pc-stats-file> is generated by running {prog} eval and passing 
        the --pc flag.
        
Options:
    -m / --metric <metric>
        Specifies what metric to evaluate prefetchers by. Currently, the 
        options are: {metric_options}

    --dry-run
        If passed, builds the traces but writes nothing.
'''.format(
        prog=sys.argv[0],
        metric_options=pc_trace.metrics
    ),

    'offline_pc_trace': '''usage: {prog} offline_pc_trace <pc-stats-file> 
                                     <output-dir> 
                                     [-m / --metric <metric>] 
                                     [-t / --num-threads <num-threads>] 
                                     [--dry-run]

Description:
    {prog} pc_trace <pc-stats-file> <pref-trace-dir>
        Parses a PC stats file, and for each PC in each trace, 
        determines the best prefetcher under <metric>. For each 
        SimPoint, it builds a new "combined" trace in <pref-trace-dir> 
        that uses the prefetches from the best-performing prefetcher on 
        each PC.
        
        <pc-stats-file> is generated by running {prog} eval and passing 
        the --pc flag.
        
Options:
    -m / --metric <metric>
        Specifies what metric to evaluate prefetchers by. Currently, the 
        options are: {metric_options}
        
    -t / --num-threads <num-threads>
        If passed (along with an argument to -d), will run <num-thread> 
        threads to perform the offline trace processing for multiple 
        benchmarks simultaneously.
        
        **CHECK YOUR SYSTEM'S MEMORY**, you may use >1 GB of memory per 
        thread.
        
    --dry-run
        If passed, builds the traces but writes nothing.
'''.format(
        prog=sys.argv[0],
        metric_options=pc_trace.metrics
    ),
}


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
    print('        # warmup inst      :',
          cfg.champsim.warmup_instructions, 'M')
    print('        track pc stats?   :', cfg.champsim.track_pc_stats)
    print('        track addr stats? :', cfg.champsim.track_addr_stats)
    print('        track prefetches? :', cfg.champsim.track_pref)
    print('        seeds              :', cfg.champsim.seeds)
    print('    Directories:')
    print('        ChampSim           :', cfg.paths.champsim_dir)
    print('        Experiment         :', cfg.paths.exp_dir)
    print('        Traces             :', cfg.paths.trace_dir)
    print('    L1D:')
    print('        Pref. candidates   :', ', '.join(cfg.l1d.pref_candidates))
    print('        Hybrids            :', cfg.l1d.hybrids)
    print('        Degrees            :', cfg.l1d.degrees)
    print('    L2:')
    print('        Pref. candidates   :', ', '.join(cfg.l2c.pref_candidates))
    print('        Hybrids            :', cfg.l2c.hybrids)
    print('        Degrees            :', cfg.l2c.degrees)
    print('    LLC:')
    print('        Sets               :', cfg.llc.sets)
    print('        Pref. candidates   :', ', '.join(cfg.llc.pref_candidates))
    print('        Replacement        :', cfg.llc.repl)
    print('        Hybrids            :', cfg.llc.hybrids)
    print('        Degrees            :', cfg.llc.degrees)
    if 'pythia' in cfg:
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
        print()
    if 'bingo' in cfg:
        print('    Bingo:')
        print('        Fixed degree       :', not cfg.bingo.dyn_degree)
    if 'spp_dev2' in cfg:
        print('    SPP_dev2:')
        print('        Fixed degree       :', not cfg.spp_dev2.dyn_degree)

    condor.build_sweep(cfg, dry_run=args.dry_run, verbose=args.verbose)


def eval_command():
    """Eval command
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
    parser.add_argument('--pc', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])

    # Add champsim_results/ to the path if it wasn't provided.
    results_dir = args.results_dir
    if not results_dir.endswith('champsim_results/'):
        results_dir = os.path.join(results_dir, 'champsim_results/')

    print('Generating cumulative run statistics...')
    evaluate.generate_run_csv(
        args.results_dir,
        args.output_file,
        weights_file=args.weight_file,
        dry_run=args.dry_run
    )

    if args.pc:
        print('Generating per-PC run statistics...')
        evaluate.generate_pc_csv(
            results_dir,
            args.output_file.replace('.csv', '_pc_llc.csv'),
            level='llc',
            dry_run=args.dry_run
        )


def best_degree_command():
    """Best Degree command

    Return the best degree for each prefetcher and trace/phase.
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('-o', '--output-file', type=str,
                        default=default_eval_csv)
    parser.add_argument('-m', '--metric', type=str,
                        default=default_best_degree_metric)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])

    # Add champsim_results/ to the path if it wasn't provided.
    results_dir = args.results_dir
    if not results_dir.endswith('champsim_results/'):
        results_dir = os.path.join(results_dir, 'champsim_results/')

    evaluate.generate_best_degree_csv(
        results_dir,
        args.output_file,
        metric=args.metric,
        dry_run=args.dry_run
    )


def online_pc_trace_command():
    """Online PC trace command
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('pc_stats_file', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-m', '--metric', type=str,
                        default=default_pc_trace_metric)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[2:])

    # Build PC traces, for evaluation online (multi_pc_trace)
    pc_trace.build_online_pc_traces(
        args.pc_stats_file,
        args.output_dir,
        args.metric,
        level='llc',
        dry_run=args.dry_run
    )


def offline_pc_trace_command():
    """Offline PC trace command
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('pc_stats_file', type=str)
    parser.add_argument('pref_traces_dir', type=str)
    parser.add_argument('-m', '--metric', type=str,
                        default=default_pc_trace_metric)
    parser.add_argument('-t', '--num_threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args(sys.argv[2:])

    # Best prefetch traces, for evaluation offline (from_file)
    pc_trace.build_offline_pc_traces(
        args.pref_traces_dir,
        args.pc_stats_file,
        args.metric,
        level='llc',
        num_threads=args.num_threads,
        dry_run=args.dry_run
    )


"""
Help
"""


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


"""
Main
"""
commands = {
    'condor': condor_command,
    'eval': eval_command,
    'best_degree': best_degree_command,
    'online_pc_trace': online_pc_trace_command,
    'offline_pc_trace': offline_pc_trace_command,
    'help': help_command,
}


def main():
    # If no subcommand specified or invalid subcommand, print main help
    # string and exit
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(help_str['help'])
        exit(-1)

    # Run specified subcommand
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
