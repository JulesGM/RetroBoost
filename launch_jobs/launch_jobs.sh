#!/usr/bin/env bash


SOURCE=${BASH_SOURCE[0]}

PYTHON="/home/mila/g/gagnonju/.main/bin/python"
RUN_CONFIGS="/home/mila/g/gagnonju/ContrievedBoosting/run_configs/"
SCRIPT_PATH="/home/mila/g/gagnonju/ContrievedBoosting/"
TARGET="$SCRIPT_PATH"run.py

if [ ! -e "$TARGET" ]; then
    echo "Config file $SCRIPT_PATH does not exist"
    exit
fi

ARRAY=(rte/baseline.json  rte/baseline_very_large.json  rte/us.json rte/us_very_large.json boolq/baseline.json  boolq/baseline_very_large.json  boolq/us.json rte/us_very_large.json)

for x in "${ARRAY[@]}" ; do
    CONFIG_PATH="$RUN_CONFIGS"$x
    if [ ! -e "$CONFIG_PATH" ]; then
        echo "Config file $CONFIG_PATH does not exist"
        continue
    fi

    


    sbatch --gres=gpu:1 --reservation=DGXA100 --mem 16GB -c 12 -n 1 "$SCRIPT_PATH"run.py "$CONFIG_PATH" "$TARGET" &
done;
