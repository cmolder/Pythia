# ChampSim Experiments

You can launch these scripts from the base repoistory directory.
- *Example*: In `Pythia/`, run `./experiments/champsim.py run <args>`.
- *Warning*: Launching these inside the `Pythia/experiments` directory may cause unexpected results, as it doesn't expect to be launched from here.

If you need help, you can run:
- `./experiments/<experiment.py> help` to get a list of commands
- `./experiments/<experiment.py> help <command>` to get help for a certain command.

---
# Prerequisites
- [Python 3](https://www.python.org/)
    - Running and building (via champsim.py) needs no additional libraries.
    - For other scripts:
        - Need Python 3.9 or older
        - Need Pandas, Numpy, Scipy, tqdm, attrdict, PyYAML libraries
- Installing dependencies (Anaconda / Miniconda):
    1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    2. [Create a Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
    3. [Activate the environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)
    4. Inside the environment, run these commands:
        - `conda install -c conda-forge "python=3.9" pandas numpy scipy tqdm`
        - `pip install attrdict PyYAML`

---
# ChampSim (champsim.py)
Contains useful commands for building specific ChampSim binaries and running them.

## champsim.py build
Builds a ChampSim binary. The L1D, L2, and LLC prefetchers are configurable, as well as the number of cores, LLC sets, branch predictor, and LLC replacement policy.

Details about prefetchers:
- **no**: No prefetcher
- **multi**: Designed so that a single binary contains all prefetchers, and the prefetchers can be chosen at runtime. Each level has a different set of prefetchers inside its "multi" prefetcher.
- **multi_pc_trace**: Similar to the "multi" prefetcher, but automatically chooses the prefetchers on a per-PC basis, using a *PC trace* that maps PCs to the prefetchers that should prefetch it. (see [prefetcher_zoo pc_trace](./README.md#prefetcher_zoopy-pc_trace) to generate PC traces). (LLC only)
- **from_file**: Issues prefetches by reading from a *prefetch address trace*, instead of an online prefetcher scheme. (Under construction, currently buggy) (LLC only)

## champsim.py run
Runs a ChampSim binary, on (a series of) ChampSim traces.

See [prefetcher_zoo condor](./README.md#prefetcher_zoopy-condor) for a list of valid prefetchers that can be passed to the `--l1d-pref`, `--l2c-pref`, and `--llc-pref` flags.

---
# Experiment configurations (exp_config/)
Defines parameters for creating the experiment sweeps below. Some parameters are specific to the experiment type, and others are used across all experiments. Example files are contained in `experiments/exp_config`.

Parameters that are common across sweeps:
- `champsim`: Specifies parameters of the simulator:
    - `warmup_instructions`: The number of warmup instructions (in millions)
    - `sim_instructions`: The number of simulation instructions, after warmup finishes (in millions)
    - `track_pc_pref`: Whether to track per-PC prefetch statistics, and save them to `<exp_dir>/pc_pref_stats`.
    - `track_addr_pref`: Whether to track per-address prefetch statistics, and save them to `<exp_dir>/per_addr_stats`.
        - *Warning*: These address statistics files can get large (10s of MB per run).
    - `branch_pred`: The branch predictor to use, defined in `branch/*.bpred`
- `l1d`, `l2c`, `llc`: Specifies parameters of each level of the cache.
    - `sets` (LLC only): The number of sets for the cache
    - `repl` (LLC only): The replacement policy for the cache
    - `pref_candidates`: A list of prefetch candidates, defined in `prefetchers/multi.<level>_pref`, that will be used within the level.
    - `max_hybrid`: The maximum hybrid to sweep over. For example, `max_hybrid: 2` will run all combinations of `pref_candiates` up to 2 running at the same time (including each candidate separately, and no prefetcher).
- `paths`: Specifies paths the experiment will need.
    - `exp_dir`: Where the outputs of the experiment, and necessary files to run the experiment, will be placed. The structure is as follows:
        - `<exp_dir>/condor/`: Condor configurations for each run
        - `<exp_dir>/scripts/`: Scripts to launch each run
        - `<exp_dir>/champsim_results/`: Saves what ChampSim prints on each run, plus (if requested) additional outputs (e.g. Per-PC stats, Per-address stats, prefetch address traces)
        - `<exp_dir>/logs/`: Output of the scripts, both stdout (.OUT) and stderr (.ERR)
        - `<exp_dir>/condor_configs_champsim.txt`: A manifest of condor files, one per line. You can [submit each job manually](https://research.cs.wisc.edu/htcondor/tutorials/intl-grid-school-3/submit_first.html), or use a script to do so in batches.
    - `trace_dir`: Where the traces reside. It expects compressed traces, with the extension `.trace.gz`.
    - `champsim_dir`: Where ChampSim resides. Use the base directory of this repository (and don't forget to build the binaries.)
- `condor`: Specifies the configuration for running on a Condor cluster. This might need to be changed, if your Condor cluster is different.
    - `user`: The user / email address of the account running the experiments.
    - `group`: The group of the account running the experiments.
    - `project`: Similar to group
    - `description`: A short summary of the experiment name.

---
# Prefetcher Zoo (prefetcher_zoo.py)
Perform an exhaustive sweep of the Prefetcher Zoo. Specifically, from a set of prefetcher candidates in each level of the cache, run a set of traces across *every* combination of these prefetchers up to a limit (say, hybrids of 2).

## prefetcher_zoo.py condor
Sets up the exhaustive sweep on a Condor cluster.

L1D prefetchers (chosen by `l1d.pref_candidates` list):
- Normal prefetchers (can by hybridized):
    - Next line: `next_line` (Untested)
    - Stride: `stride` (Untested)
    - IPCP: `ipcp`  (Untested)
    
L2C prefetchers (chosen by `l2c.pref_candidates` list):
- Normal prefetchers (can by hybridized):
    - SMS: `sms`  (Untested)
    - Pythia: `scooby`
    - Next-line: `next_line` (Untested)
    - Best Offset: `bop`  (Untested)
    - Sandbox: `sandbox` (Untested)
    - Dspatch: `dspatch` (Untested)
    - SPP: `spp_dev2`  (Untested)
    - SPP+PPF: `ppf_dev` (Untested)
    - MLOP: `mlop` (Untested)
    - Bingo: `bingo`  (Untested)
    - Stride: `stride` (Untested)
    - AMPM: `ampm` (Untested)
    - Streamer: `streamer` (Untested)

LLC prefetchers (chosen by `llc.pref_candidates` list):
- Normal prefetchers (can by hybridized):
    - SMS: `sms`
    - Pythia: `scooby`
    - Next-line: `next_line` (Untested)
    - Best Offset: `bop`
    - Sandbox: `sandbox` (Untested)
    - Dspatch: `dspatch` (Untested)
    - SPP: `spp_dev2`
    - SPP+PPF: `ppf_dev` (Untested)
    - MLOP: `mlop` (Currently broken)
    - Bingo: `bingo`
    - Stride: `stride` (Untested)
    - AMPM: `ampm` (Untested)
    - Streamer: `streamer` (Untested)
    - Idealized ISB: `sisb`
    - Triage: `triage` (Currently broken)
- Special prefetchers (cannot be hybridzed)
    - Run prefetchers online, from best PC trace: `pc_trace`
        - Runs `multi_pc_trace.llc_pref`, which using a Best PC trace (see [prefetcher_zoo pc_trace](./README.md#prefetcher_zoopy-pc_trace)), automatically selects the best performing prefetcher(s) and degrees to prefetch each load PC's loads.
        - Need to build `multi_pc_trace` separately (pass `--llc-pref multi_pc_trace` to `champsim.py build`)
    - Run prefetchers offline, from prefetch address trace: `from_file`
        - Runs `from_file.llc_pref`, which prefetches using a per-instruction trace that lists the addresses and levels to prefetch.
        - Need to build `from_file` seprately (pass `--llc-pref from_file` to `champsim.py build`)
        
Inside the config file, the option `paths.degree_csv` can be passed to select a degree for each prefetcher combination on each simpoint. This can be obtained from the output of [prefetcher_degree_sweep eval](./README.md#prefetcher_degree_sweeppy-eval). If it is not provided, default degrees are used (defined in the default knobs file / in `src/knobs.cc`).
    

## prefetcher_zoo.py eval
Gathers statitics on each run, saving each run's IPC, per-level MPKI, per-level accuracy/coverage, and improvement over no prefetching, and more to a single .csv file.

## prefetcher_zoo.py pc_trace
Builds a Best PC trace for each trace (simpoint). A Best PC trace lists the best-performing prefetcher on each PC of the trace, under some metric. A Best PC trace can be fed into a `multi_pc_trace` prefetcher to evaluate the selections online. 

*Note*: To gather this data, the configuration variable `champsim.track_pc_pref` must be true in the original sweep.

Metrics (chosen by `-m / --metric` flag):
- `num_useful`: The number of useful prefetches on that PC
- `marginal_useful`: The number of useful prefetches on that PC, minus the number of useless prefetches
- `accuracy`: The accuracy of the prefetcher on that PC.

---
# Degree Sweep (prefetcher_degree_sweep.py)
Perform an exhaustive sweep of the prefetcher zoo, but *also vary* each constitutent prefetcher's degree.

Inside the config file, the field `max_degree` inside each cache level controls the maximum degree for prefetchers on that level. For example, `max_degree 8` sweeps all degrees from 1 to 8 for each constituent prefetcher in each hybrid (at that cache level). 

Some prefechers (e.g. Pythia, SPP, Bingo) manage their degree automatically, so they are not swept on degree.

## prefetcher_degree_sweep.py condor

TODO: Description

## prefetcher_degree_sweep.py eval

TODO: Description

---
# Pythia Level (pythia_level.py)
Sweep on a variant of Pythia that prefetches into the LLC (instead of L2) when its confidence in a prefetch candidate is low. 

This is done by comparing the Q-value to a threshold, and issuing the prefetch to the LLC instead of L2 if the Q-value is below that threshold.

## pythia_level.py condor

TODO: Description

## pythia_level.py eval

TODO: Description

---
# A good workflow
This is a good workflow to follow, assuming sufficient computational resources:
1. Create a degree sweep over the set of prefetchers, using [prefetcher_degree_sweep.py condor](#prefetcher_degree_sweep.py-condor)
2. Run the degree sweep.
3. Evaluate the degree sweep to get the best degree for each prefetcher combination, using [prefetcher_degree_sweep.py eval](#prefetcher_degree_sweep.py-eval)
3. Create a zoo sweep, filtering on the best degrees from the degree sweep, using [prefetcher_zoo.py condor](#prefetcher_zoo.py-condor) with `path.degree_csv` inside the sweep config file.
    - The idea of doing another zoo sweep separately is to reduce the amount of extraneous per-PC statistics, per-address statistics, and trace files generated, since we are only considering the best degree.
4. Run the zoo sweep.
5. Evaluate the zoo sweep to get results for each (tuned) prefetcher combination, using [prefetcher_zoo.py eval](#prefetcher_zoo.py-eval).

If you want to evaluate *online* PC-localized headroom: (on the LLC)
1. Do all the above.
2. Run [prefetcher_zoo.py pc_trace](#prefetcher_zoo.py-pc_trace) to get the Best PC trace for each simpoint (on a metric).
3. Create another zoo sweep, using `pc_trace` as the prefetcher (Example: `experiments/exp_config/zoo_pctrace_num_useful.yml`)
4. Run the second zoo sweep.
5. Evaluate the second zoo sweep to get results for each simpoint on its Online Best PC combination.

If you want to evaluate *offline* PC-localized headroom: (on the LLC, under construction!)
1. Do all the above
2. Create another zoo sweep, using `from_file` as the prefetcher. (TODO: Add directory for address traces)
3. Run the second zoo sweep.
4. Evaluate the second zoo sweep to get results for each simpoint on its Offline Best PC combination.
