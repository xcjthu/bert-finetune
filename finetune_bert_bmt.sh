#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="./"
#VERSION="bert-base-cased"
VERSION="/data/home/scv0540/.cache/model_center/bert-base-cased/"
DATASET="BoolQ"

OPTS=""
OPTS+=" --model-config ${VERSION}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset_name ${DATASET}"
OPTS+=" --batch-size 56"
OPTS+=" --lr 0.0001"
OPTS+=" --max-decoder-length 512"
OPTS+=" --train-iters 1400"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --loss-scale 128"

CMD="python3 >${LOGFILE} -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_bert_bmt.py ${OPTS} ${CMDOPTS}"
echo ${CMD}

python3 -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_bert_bmt.py ${OPTS} ${CMDOPTS} 2>&1 | tee ${LOGFILE}
