from llm_utils.api import Openai_api_handler
import argparse
from utils.mydataset import RareDataset
from utils.evaluation import diagnosis_evaluate
import os
from prompt import RarePrompt
import json
import numpy as np
import re
from benchflow import BenchClient
from typing import Dict, Any

np.random.seed(42)

class RareBenchClient(BenchClient):
    def __init__(self, intelligence_url: str, max_retry: int = 1):
        super().__init__(intelligence_url, max_retry)

    def prepare_input(self, raw_input_data: Dict[str, Any]) -> Dict[str, Any]:
        return raw_input_data

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        result = {
                    'system_prompt': "",
                    'question': "",
                    'model': "user_model",
                    'seed': 42,
                    'usage': {
                        'input_tokens': 0,
                        'output_tokens': 0,
                    },
                    'answer': raw_response,
                }
        return result

def diagnosis_metric_calculate(folder, judge_model="chatgpt"):
    handler = Openai_api_handler(judge_model)
    
    CNT = 0
    metric = {}
    recall_top_k = []

    Pediatrics = range(0, 15)
    Neurology = range(30, 45)
    Cardiology = range(15, 30)
    Nephrology = range(45, 60)
    Hematology = range(60, 75)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        res = json.load(open(file_path, "r", encoding="utf-8-sig"))
        predict_rank = res["predict_rank"]
        if res['predict_diagnosis'] is None:
            print(file_path, "predict_diagnosis is None")
        
        if predict_rank is None:
            predict_rank = diagnosis_evaluate(res["predict_diagnosis"], res["golden_diagnosis"], handler)
            res["predict_rank"] = predict_rank
            json.dump(res, open(file_path, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
        
        if predict_rank not in ["否", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "No"]:
            print(file_path)
            CNT += 1

        if "否" in predict_rank or "No" in predict_rank:
            recall_top_k.append(11)
        else:
            pattern = r'\b(?:10|[1-9])\b'
            found = re.findall(pattern, predict_rank)
            if not found or found[0] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
                res["predict_rank"] = None
                raise Exception("predict_rank error")
            predict_rank = found[0]
            recall_top_k.append(int(predict_rank))
    
    metric['recall_top_1'] = len([i for i in recall_top_k if i <= 1]) / len(recall_top_k)
    metric['recall_top_3'] = len([i for i in recall_top_k if i <= 3]) / len(recall_top_k)
    metric['recall_top_10'] = len([i for i in recall_top_k if i <= 10]) / len(recall_top_k)
    metric['medain_rank'] = np.median(recall_top_k)
    
    print(folder)
    print(metric)
    print("predict_rank error: ", CNT)
    print("evaluate tokens: ", handler.gpt4_tokens, handler.chatgpt_tokens, handler.chatgpt_instruct_tokens)
    
    # Package the results into a dictionary
    result = {
        "folder": folder,
        "metric": metric,
        "predict_rank_error": CNT,
    }
    
    # Save the results to result.json in the folder
    result_file = os.path.join("./results", "result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
def generate_random_few_shot_id(exclude_id, total_num, k_shot=3):
    few_shot_id = []
    while len(few_shot_id) < k_shot:
        id = np.random.randint(0, total_num)
        if id not in few_shot_id and id not in exclude_id:
            few_shot_id.append(id)
    return few_shot_id

def generate_dynamic_few_shot_id(methods, exclude_id, dataset, k_shot=3):
    few_shot_id = []

    patient = dataset.load_hpo_code_data()
    if methods == "dynamic":
        phe2embedding = json.load(open("mapping/phe2embedding.json", "r", encoding="utf-8-sig"))
    elif methods == "medprompt":
        phe2embedding = json.load(open("mapping/medprompt_emb.json", "r", encoding="utf-8-sig"))
    ic_dict = json.load(open("mapping/ic_dict.json", "r", encoding="utf-8-sig"))
    if methods == "medprompt":
        ic_dict = {k: 1 for k, _ in ic_dict.items()}
   
    exclude_patient = patient[exclude_id]
    exclude_patient_embedding = np.array([np.array(phe2embedding[phe]) for phe in exclude_patient[0] if phe in phe2embedding])
    exclude_patient_ic = np.array([ic_dict[phe] for phe in exclude_patient[0] if phe in phe2embedding])
    exclude_patient_embedding = np.sum(exclude_patient_embedding * exclude_patient_ic.reshape(-1, 1), axis=0) / np.sum(exclude_patient_ic)
    candidata_embedding_list = []
    for i, p in enumerate(patient):
        phe_embedding = np.array([np.array(phe2embedding[phe]) for phe in p[0] if phe in phe2embedding])
        ic_coefficient_list = np.array([ic_dict[phe] for phe in p[0] if phe in phe2embedding])
        phe_embedding = np.sum(phe_embedding * ic_coefficient_list.reshape(-1, 1), axis=0) / np.sum(ic_coefficient_list)
        candidata_embedding_list.append(phe_embedding)
    candidata_embedding_list = np.array(candidata_embedding_list)
    cosine_sim = np.dot(candidata_embedding_list, exclude_patient_embedding) 
    cosine_sim = np.argsort(cosine_sim)[::-1]
    for i in cosine_sim:
        if i not in few_shot_id and i != exclude_id:
            few_shot_id.append(i)
        if len(few_shot_id) == k_shot:
            break
    
    return few_shot_id


def run_task(task_type, dataset:RareDataset, intelligence_url, results_folder, few_shot, cot, judge_model, eval=False):
    few_shot_dict = {}
    rare_prompt = RarePrompt()
    client = RareBenchClient(intelligence_url)
    if task_type == "diagnosis":
        patient_info_type = dataset.dataset_type
        os.makedirs(results_folder, exist_ok=True)
        print("Begin diagnosis.....")
        print("total patient: ", len(dataset.patient))
        ERR_CNT = 0
        questions = []
        for i, patient in enumerate(dataset.patient):
            # if handler is None:
            #     print("handler is None")
            #     break
            result_file = os.path.join(results_folder, f"patient_{i}.json")
            if os.path.exists(result_file):
                continue
            patient_info = patient[0]
            golden_diagnosis = patient[1]
            few_shot_info = []
            if few_shot == "random":
                few_shot_id = generate_random_few_shot_id([i], len(dataset.patient))
                
                few_shot_dict[i] = few_shot_id
                for id in few_shot_id:
                    few_shot_info.append((dataset.patient[id][0], dataset.patient[id][1]))
            elif few_shot == "dynamic" or few_shot == "medprompt":
                few_shot_id = generate_dynamic_few_shot_id(few_shot, i, dataset)
                
                few_shot_dict[str(i)] = [str(idx) for idx in few_shot_id]
                for id in few_shot_id:
                    few_shot_info.append((dataset.patient[id][0], dataset.patient[id][1]))

            system_prompt, prompt = rare_prompt.diagnosis_prompt(patient_info_type, patient_info, cot, few_shot_info)
            
            questions.append(system_prompt + prompt)
            # if few_shot == "auto-cot":
            #     autocot_example = json.load(open("mapping/autocot_example.json", "r", encoding="utf-8-sig"))
            #     system_prompt = "Here a some examples: " + autocot_example[handler.model_name] + system_prompt
            #     prompt = prompt + "Let us think step by step.\n"
            

            # predict_diagnosis = handler.get_completion(system_prompt, prompt)
            input = {"system_prompt": system_prompt, "prompt": prompt}
            predict_diagnosis = client.get_response(input)
            if predict_diagnosis is None:
                print(f"patient {i} predict diagnosis is None")
                ERR_CNT += 1
                continue
            
            predict_rank = None
            res = {
                "patient_info": patient_info,
                "golden_diagnosis": golden_diagnosis,
                "predict_diagnosis": predict_diagnosis['raw_prediction'],
                "predict_rank": predict_rank
            }
            json.dump(res, open(result_file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
            print(f"patient {i} finished")
            # if type(handler) == Openai_api_handler:
            #     print("total tokens: ", handler.gpt4_tokens, handler.chatgpt_tokens, handler.chatgpt_instruct_tokens)
            
        if eval:
            diagnosis_metric_calculate(results_folder, judge_model=judge_model)
        print("diagnosis ERR_CNT: ", ERR_CNT)
    elif task_type == "mdt":
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intelligence_url', type=str, default="http://localhost:8000")
    parser.add_argument('--task_type', type=str, default="diagnosis", choices=["diagnosis", "mdt"])
    parser.add_argument('--dataset_name', type=str, default="LIRICAL", choices=["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM", "PHENOTYPE"])
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--results_folder', default='./results/PUMCH')
    parser.add_argument('--judge_model', type=str, default="chatgpt", choices=["gpt4", "chatgpt"])
    parser.add_argument('--few_shot', type=str, default="none", choices=["none", "random", "dynamic", "medprompt"])
    parser.add_argument('--cot', type=str, default="none", choices=["none", "zero-shot"])
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    # if args.model in ["gpt4", "chatgpt"]:
    #     handler = Openai_api_handler(args.model)

    dataset = RareDataset(args.dataset_name, args.dataset_path)
    
    if args.few_shot == "none":
        few_shot = ""
    elif args.few_shot == "random":
        few_shot = "_few_shot"
    elif args.few_shot == "dynamic":
        few_shot = "_dynamic_few_shot"
    elif args.few_shot == "medprompt":
        few_shot = "_medprompt"
    elif args.few_shot == "auto-cot":
        few_shot = "_auto-cot"
    if args.cot == "none":
        cot = ""
    elif args.cot == "zero-shot":
        cot = "_cot"
    results_folder = os.path.join(args.results_folder, args.dataset_name, args.task_type+few_shot+cot)
    run_task(args.task_type, dataset, args.intelligence_url, results_folder, args.few_shot, args.cot, args.judge_model, args.eval)

if __name__ == "__main__":
    main()