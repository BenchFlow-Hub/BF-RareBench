#! /bin/bash

INTELLIGENCE_URL=${INTELLIGENCE_URL:-${AGENT_URL}}
DATASET_NAME=${TEST_START_IDX:-"LIRICAL"}
DATASET_TYPE=${DATASET_TYPE:-"PHENOTYPE"}
JUDGE_MODEL=${JUDGE_MODEL:-"chatgpt"}
FEW_SHOT=${FEW_SHOT:-"none"}
COT=${COT:-"none"}

echo "running with $INTELLIGENCE_URL, $DATASET_NAME, $DATASET_TYPE, $JUDGE_MODEL, $FEW_SHOT, $COT"
python main.py --intelligence_url=$INTELLIGENCE_URL --task_type=diagnosis --dataset_name=$DATASET_NAME --dataset_type=$DATASET_TYPE --judge_model=$JUDGE_MODEL --few_shot=$FEW_SHOT --cot=$COT --eval
