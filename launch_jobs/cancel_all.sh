#! /usr/bin/env bash
python -c "
import subprocess
op = subprocess.check_output('squeue -u gagnonju', shell=True).decode().strip();
lines = op.split('\n')[1:]
for line in lines:
    entries = line.split()[0]
    print(entries)
"  | xargs scancel
