#! /bin/bash

INTELLIGENCE_URL=${INTELLIGENCE_URL:-"http://localhost:8000"}
TASK_TYPE=${TASK_TYPE:-"diagnosis"}
DATASET_NAME=${DATASET_NAME:-"LIRICAL"}
DATASET_TYPE=${DATASET_TYPE:-"PHENOTYPE"}
JUDGE_MODEL=${JUDGE_MODEL:-"chatgpt"}
FEW_SHOT=${FEW_SHOT:-"none"}
COT=${COT:-"none"}

python main.py --intelligence_url=$INTELLIGENCE_URL --task_type=$TASK_TYPE --dataset_name=$DATASET_NAME --dataset_type=$DATASET_TYPE --judge_model=$JUDGE_MODEL --few_shot=$FEW_SHOT --cot=$COT --eval
