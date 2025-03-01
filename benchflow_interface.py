import os
from typing import Any, Dict
import json
from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class RareBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_args(self, task_id: str) -> BenchArgs:
        arguments = {
            "required": ["OPENAI_API_KEY"],
            "optional": [
                {"DATASET_TYPE": "PHENOTYPE"},
                {"JUDGE_MODEL": "chatgpt"},
                {"FEW_SHOT": "none"},
                {"COT": "none"},
                {"TEST_END_IDX": f"{task_id}"}
            ],
        }
        return BenchArgs(arguments)
    
    def get_image_name(self) -> str:
        """
        Return the Docker image name for running the WebArena benchmark.
        """
        return "kirk2000/benchflow:rarebench-v1"
    
    def get_results_dir_in_container(self) -> str:
        """
        Return the directory inside the container where the benchmark results will be stored.
        """
        return "/app/results"
    
    def get_log_files_dir_in_container(self) -> str:
        """
        Return the directory inside the container where the log files will be stored.
        """
        return "/app/logs"
    
    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        Read and parse the benchmark result from the log files.
        
        This method expects a file named 'log_files.txt' in the results directory.
        It then reads the content of each log file listed in 'log_files.txt',
        aggregates the log output, and extracts the average score and pass status.
        """
        results_txt = os.path.join(self.results_dir, "result.json")
        if not os.path.exists(results_txt):
            return BenchmarkResult(task_id=task_id, is_resolved=False, metrics={"score": 0},log={"error": "No results found"}, other={})
        
        log_content = ""
        try:
            with open(results_txt, 'r') as f:
                result = json.load(f)
        except Exception as e:
            return BenchmarkResult(task_id=task_id, is_resolved=False, metrics={"score": 0}, log={"error": f"Failed to read log files: {e}"}, other={})
        
        # Parse the log content to extract score and status
        is_resolved = True
        metrics = result["metric"]
        log_content_dir = os.path.join(self.results_dir, os.path.relpath(result["folder"], "./results"))
        for file in os.listdir(log_content_dir):
            with open(os.path.join(log_content_dir, file), 'r') as f:
                log_content += f.read() + "\n"
        return BenchmarkResult(task_id=task_id, is_resolved=is_resolved, metrics=metrics, log={"details": log_content}, other={})
    
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return a dictionary with all task IDs and an optional error message.
        For 'train' split, return 200 tasks; otherwise, return 812 tasks.
        """
        return {"task_ids": ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM", "PHENOTYPE"], "error_message": None}
    
    def cleanup(self):
        """
        Clean up benchmark resources by removing the local results and log files directories.
        """
        pass