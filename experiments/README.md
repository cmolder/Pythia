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
- **multi_pc_trace**: Similar to the "multi" prefetcher, but automatically chooses the prefetchers on a per-PC basis, using a *PC trace* that maps PCs to the prefetchers that should prefetch it. (see prefetcher_zoo.py pc_trace to generate PC traces). (LLC only)
- **from_file**: Issues prefetches by reading from a *prefetch address trace*, instead of an online prefetcher scheme. (Under construction, currently buggy) (LLC only)

## champsim.py run

TODO: Description

---
# Experiment configurations (exp_config/)
Defines parameters for creating the experiment sweeps below. Some parameters are specific to the experiment type, and others are used across all experiments.

TODO: Description

---
# Prefetcher Zoo (prefetcher_zoo.py)

TODO: Description

## prefetcher_zoo.py condor

TODO: Description

## prefetcher_zoo.py eval

TODO: Description

## prefetcher_zoo.py pc_trace

TODO: Description

---
# Degree Sweep (prefetcher_degree_sweep.py)

TODO: Description

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
