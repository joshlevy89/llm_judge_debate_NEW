import os
import time
from pathlib import Path
from dotenv import load_dotenv
import config.config_qa as config_qa
from config.config_qa import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    MODEL_NAME, TEMPERATURE,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES,
    SPECIFIC_QUESTION_IDXS, MAX_THREADS, RERUN
)
from datasets import load_dataset
from utils.dataset_utils import select_questions_and_options
from utils.shared_utils import extract_config
from utils.qa_utils import run_qa_for_questions, get_existing_qa_keys, filter_existing_questions

def main():
    load_dotenv()    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    results_path = Path('results') / 'qa' / 'qa_results.jsonl'
    dataset_config = extract_config(config_qa)
    
    print(f"Results will be appended to: {results_path}")
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    
    if SPECIFIC_QUESTION_IDXS is not None:
        print(f"Using specific question indices: {SPECIFIC_QUESTION_IDXS}")
        question_idxs = SPECIFIC_QUESTION_IDXS
    else:
        print(f"Selecting {NUM_QUESTIONS} questions with seed {RANDOM_SEED}")
        import random
        rng = random.Random(RANDOM_SEED)
        question_idxs = rng.sample(range(len(dataset)), min(NUM_QUESTIONS, len(dataset)))
    
    if not RERUN:
        existing_qa = get_existing_qa_keys(results_path)
        questions_data = select_questions_and_options(DATASET_NAME, dataset, len(question_idxs), NUM_CHOICES, None, question_idxs)
        question_idxs = filter_existing_questions(question_idxs, questions_data, MODEL_NAME, NUM_CHOICES, existing_qa)
        
        if not question_idxs:
            print("All questions already have QA results. Nothing to run. Can set RERUN = True to rerun.")
            return
        
        print(f"Filtered to {len(question_idxs)} questions without existing results")
    
    start_time = time.time()
    print(f"Processing {len(question_idxs)} questions...")
    
    result = run_qa_for_questions(
        question_idxs=question_idxs,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        dataset_config=dataset_config,
        num_choices=NUM_CHOICES,
        api_key=api_key,
        max_threads=MAX_THREADS,
        qa_results_path=results_path,
        random_seed=RANDOM_SEED
    )
    
    duration = time.time() - start_time
    
    print(f"\nRun ID: {result['run_id']}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: total {len(question_idxs)}, success {result['completed']}, error {result['failed']}")
    print(f"Cost: ${result['cost']:.6f}")

if __name__ == "__main__":
    main()
