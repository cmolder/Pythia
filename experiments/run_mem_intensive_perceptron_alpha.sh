#!/bin/bash

# Check that args are correct
if [ "$#" -ne 1 ]; then
    echo "Incorrect # of arguments: expected cache replacement policy"
    echo "Usage: ./run_cluster.sh name_of_policy"
    exit
fi

# Check that we are logged into streetpizza
#HOST="$(hostname)"
#if [ "${HOST}" != "jalad" && "${HOST}" != "azog"]; then
#    echo "You must be logged into jalad.cs.utexas.edu for this script to work!"
#    echo "First, ssh utcsid@jalad.cs.utexas.edu, then run the script."
#    exit
#fi

# Simulation parameters
GPU=false
TRACE_PATH="/scratch/cluster/zshi17/traces"
CHAMP_PATH="$(pwd)"
PREFETCH="scooby"
ALPHA=${1}
OUTPUT_DIR="${CHAMP_PATH}/out/scooby_perceptron_alpha_${ALPHA}"
OUTPUT_CONDOR_DIR="${OUTPUT_DIR}/condor"

# Ensure output dir exists
if test ! -d ${OUTPUT_DIR}; then                                                                                            mkdir ${OUTPUT_DIR}
fi

# Ensure output dir exists
if test ! -d ${OUTPUT_CONDOR_DIR}; then
    mkdir ${OUTPUT_CONDOR_DIR}
fi

# Run our cache replacement policy on each trace
while read TRACE; do
    SCRIPT_FILE="${OUTPUT_CONDOR_DIR}/${TRACE}.sh"
    CONDOR_FILE="${OUTPUT_CONDOR_DIR}/${TRACE}.condor"
    
    # create script file
    echo "#!/bin/bash" > ${SCRIPT_FILE}
    echo "cd ${CHAMP_PATH}" >> ${SCRIPT_FILE}
    echo "./experiments/champsim.py run ${TRACE_PATH}/${TRACE} --l2c-pref ${PREFETCH} --results-dir ${OUTPUT_DIR} --run-name ${TRACE} --extra-knobs \"‘--le_featurewise_pooling_type=1 --scooby_gamma=0.0 --scooby_enable_featurewise_engine=false --scooby_alpha=0.${ALPHA}’\"" >> ${SCRIPT_FILE}
    chmod +x ${SCRIPT_FILE}

    # create condor file
    ~/ChampSim/condorize.sh ${GPU} ${OUTPUT_CONDOR_DIR} ${TRACE}
    
    # submit the condor file
    /lusr/opt/condor/bin/condor_submit ${CONDOR_FILE}
done < sim_list/traces_mem_intensive.txt
