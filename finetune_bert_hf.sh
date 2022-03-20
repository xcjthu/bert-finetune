#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="./"
VERSION="bert-base-cased"
DATASET="ReCoRD"

OPTS=""
OPTS+=" --model-version ${VERSION}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset_name ${DATASET}"
OPTS+=" --batch-size 8"
OPTS+=" --grad-accumulation 8"
OPTS+=" --lr 0.00001"
OPTS+=" --max-length 512"
OPTS+=" --train-iters 1400"
OPTS+=" --weight-decay 1e-2"

CMD="python3 >${LOGFILE} -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_bert_hf.py ${OPTS} ${CMDOPTS}"
echo ${CMD}

python3 -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_bert_hf.py ${OPTS} ${CMDOPTS} 2>&1 | tee ${LOGFILE}
