import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from run_verdict import main
from utils.shared_utils import generate_run_id
from utils.llm_utils import get_openrouter_key_info

JUDGE_MODELS = [
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o-mini",
    # "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
#     # "qwen/qwen3-8b",
#     # "qwen/qwen3-14b",
#     # "qwen/qwen3-32b",
#     # "x-ai/grok-4-fast",
]

# JUDGE_MODELS = [
#     "x-ai/grok-4-fast",
# ]

# DEBATE_RUN_IDS = [
    # "egkyot4",
    # "ts9ga4y",
    # "txd06z5",
    # "wcsck4w",
    # "yn1vu8h",
# ]

# DEBATE_RUN_IDS = [
#     "uveal9q",
#     "z42o1e7",
#     "79t2rwe",
#     "2exxeqn", # expensive!
#     "xcmiu00",
#     "3ys5csf",
#     "pciywxv"
# ]

DEBATE_RUN_IDS = [
    "pbbjuor"
]

# UPTO_TURNS = [0, 1, 2, 4, 6, 8, 10, 11, 12]
# UPTO_TURNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# UPTO_TURNS = [1, 2, 3, 4]
# UPTO_TURNS = [2]
# UPTO_TURNS=None
# UPTO_TURNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# UPTO_TURNS = [2, 8, 14, 20, 22]
# UPTO_TURNS = [2, 4, 6, 8]
# UPTO_TURNS = None
# UPTO_TURNS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
UPTO_TURNS = None




RUNS_PER_COMBINATION = 1
MAX_PARALLEL_PROCESSES = 6
MAX_THREADS_PER_COMBO = 40

def run_combination(args):
    """Run a single (model, debate, upto_turns, rep) combination."""
    model, debate_run_id, upto_turns, run_idx = args
    verdict_run_id = main(judge_model=model, debate_run_id=debate_run_id, upto_turns=upto_turns, max_threads=MAX_THREADS_PER_COMBO)
    return {
        "model": model,
        "debate_run_id": debate_run_id,
        "upto_turns": upto_turns,
        "run_idx": run_idx,
        "verdict_run_id": verdict_run_id
    }

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    group_run_id = generate_run_id()
    
    upto_turns_list = [None] if UPTO_TURNS is None else UPTO_TURNS
    
    combinations = [
        (model, debate_run_id, upto_turns, run_idx)
        for run_idx in range(RUNS_PER_COMBINATION)
        for upto_turns in upto_turns_list
        for debate_run_id in DEBATE_RUN_IDS
        for model in JUDGE_MODELS
    ]
    total_runs = len(combinations)
    
    print(f"Group Run ID: {group_run_id}")
    print(f"Total combinations: {total_runs} ({len(JUDGE_MODELS)} models x {len(DEBATE_RUN_IDS)} debates x {len(upto_turns_list)} turn limits x {RUNS_PER_COMBINATION} reps)")
    print(f"Parallelism: {MAX_PARALLEL_PROCESSES} processes, {MAX_THREADS_PER_COMBO} threads each")
    
    start_time = time.time()
    all_runs = []
    model_progress = defaultdict(lambda: {"completed": 0, "total": len(DEBATE_RUN_IDS) * len(upto_turns_list) * RUNS_PER_COMBINATION})
    
    executor = ProcessPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES)
    future_to_combo = {executor.submit(run_combination, combo): combo for combo in combinations}
    
    try:
        for future in as_completed(future_to_combo):
            combo = future_to_combo[future]
            model, debate_run_id, upto_turns, run_idx = combo
            try:
                result = future.result()
                all_runs.append(result)
                model_progress[model]["completed"] += 1
                prog = model_progress[model]
                key_info = get_openrouter_key_info(api_key)
                cost_str = f"${key_info['data']['usage']:.2f}" if key_info else "N/A"
                print(f"\n{'='*80}\n[BATCH] {model} | {debate_run_id} | upto_turns={upto_turns} rep {run_idx} | {prog['completed']}/{prog['total']} done | verdict: {result['verdict_run_id']} | total cost: {cost_str}\n{'='*80}\n")
            except Exception as e:
                print(f"\n{'='*80}\n[BATCH ERROR] {model} | {debate_run_id} | upto_turns={upto_turns} rep {run_idx} | {str(e)[:100]}\n{'='*80}\n")
    except KeyboardInterrupt:
        print("\n\nInterrupted! Killing child processes...")
        for pid in executor._processes:
            executor._processes[pid].terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        import sys
        sys.exit(1)
    
    duration = time.time() - start_time
    
    group_data = {
        "group_run_id": group_run_id,
        "datetime": datetime.now().isoformat(),
        "duration_seconds": duration,
        "runs_per_combination": RUNS_PER_COMBINATION,
        "judge_models": JUDGE_MODELS,
        "debate_run_ids": DEBATE_RUN_IDS,
        "upto_turns": UPTO_TURNS,
        "runs": all_runs
    }
    
    Path('results/verdict_groups').mkdir(parents=True, exist_ok=True)
    with open(f'results/verdict_groups/{group_run_id}.json', 'w') as f:
        json.dump(group_data, f, indent=2)
    
    print(f"\nGroup saved: results/verdict_groups/{group_run_id}.json")
    print(f"Duration: {duration:.1f}s")
    print(f"Completed: {len(all_runs)}/{total_runs}")
    
    executor.shutdown(wait=True)

