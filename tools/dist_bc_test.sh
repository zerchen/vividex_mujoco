#!/bin/bash
NUM_PROC=$1
MASTER_PORT=$2
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $MASTER_PORT dist_bc_test.py ${@:2}
