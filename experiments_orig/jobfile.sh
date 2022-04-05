#!/bin/bash
#
#
#
# Traces:
#    astar_163B
#    astar_23B
#    astar_313B
#
#
# Experiments:
#    nopref: --warmup_instructions=1000000 --simulation_instructions=50000000 --config=$(PYTHIA_HOME)/config/nopref.ini
#    bingo: --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bingo --config=$(PYTHIA_HOME)/config/bingo.ini
#    bo: --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bop
#    pythia: --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=scooby --config=$(PYTHIA_HOME)/config/pythia.ini
#    sms: --llc_prefetcher_types=sms --llc_prefetcher_types=sms
#    spp: --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=spp_dev2 --config=$(PYTHIA_HOME)/config/spp_dev2.ini
#
#
#
#
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --config=/u/cmolder/GitHub/Pythia/config/nopref.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_163B.trace.gz > astar_163B_nopref.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bingo --config=/u/cmolder/GitHub/Pythia/config/bingo.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_163B.trace.gz > astar_163B_bingo.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bop  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_163B.trace.gz > astar_163B_bo.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=scooby --config=/u/cmolder/GitHub/Pythia/config/pythia.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_163B.trace.gz > astar_163B_pythia.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --llc_prefetcher_types=sms --llc_prefetcher_types=sms  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_163B.trace.gz > astar_163B_sms.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=spp_dev2 --config=/u/cmolder/GitHub/Pythia/config/spp_dev2.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_163B.trace.gz > astar_163B_spp.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --config=/u/cmolder/GitHub/Pythia/config/nopref.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_23B.trace.gz > astar_23B_nopref.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bingo --config=/u/cmolder/GitHub/Pythia/config/bingo.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_23B.trace.gz > astar_23B_bingo.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bop  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_23B.trace.gz > astar_23B_bo.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=scooby --config=/u/cmolder/GitHub/Pythia/config/pythia.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_23B.trace.gz > astar_23B_pythia.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --llc_prefetcher_types=sms --llc_prefetcher_types=sms  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_23B.trace.gz > astar_23B_sms.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=spp_dev2 --config=/u/cmolder/GitHub/Pythia/config/spp_dev2.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_23B.trace.gz > astar_23B_spp.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --config=/u/cmolder/GitHub/Pythia/config/nopref.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_313B.trace.gz > astar_313B_nopref.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bingo --config=/u/cmolder/GitHub/Pythia/config/bingo.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_313B.trace.gz > astar_313B_bingo.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=bop  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_313B.trace.gz > astar_313B_bo.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=scooby --config=/u/cmolder/GitHub/Pythia/config/pythia.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_313B.trace.gz > astar_313B_pythia.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --llc_prefetcher_types=sms --llc_prefetcher_types=sms  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_313B.trace.gz > astar_313B_sms.out 2>&1
/u/cmolder/GitHub/Pythia/bin/perceptron-no-no-multi-ship-1core --warmup_instructions=1000000 --simulation_instructions=50000000 --llc_prefetcher_types=spp_dev2 --config=/u/cmolder/GitHub/Pythia/config/spp_dev2.ini  -traces /scratch/cluster/cmolder/zhan_traces/champsim_prefetching_zoo/astar_313B.trace.gz > astar_313B_spp.out 2>&1
