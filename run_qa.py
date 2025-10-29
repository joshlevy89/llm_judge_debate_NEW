import os
import re
import json
import yaml
import random
import string
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
from config_qa import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    MODEL_NAME, TEMPERATURE,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES,
    MAX_THREADS
)
from llm_utils import call_openrouter, get_openrouter_key_info, parse_model_response
from dataset_utils import select_questions_and_options, format_options

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def setup_output_path():
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / 'qa_results.jsonl'

def get_config():
    return {
        'dataset_name': DATASET_NAME,
        'dataset_subset': DATASET_SUBSET,
        'dataset_split': DATASET_SPLIT,
        'model_name': MODEL_NAME,
        'temperature': TEMPERATURE,
        'num_questions': NUM_QUESTIONS,
        'random_seed': RANDOM_SEED,
        'num_choices': NUM_CHOICES,
        'max_threads': MAX_THREADS
    }

def load_prompt_template():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['qa_prompt_template'], prompts['response_format_prompt']

def process_question(q_data, prompt_template, response_format_prompt, prompt_template_str, api_key, question_num, total_questions, config, run_id, run_datetime):
    record_id = generate_run_id()
    options_text = format_options(q_data['options'])
    number_choices = ', '.join(str(i) for i in range(NUM_CHOICES))
    
    prompt = prompt_template.format(
        question=q_data['question'],
        options_text=options_text,
        letter_choices=number_choices,
        response_format_prompt=response_format_prompt
    )
    
    response = call_openrouter(prompt, MODEL_NAME, api_key, TEMPERATURE)
    
    if 'choices' in response and len(response['choices']) > 0:
        raw_model_response = response['choices'][0]['message']['content']
    else:
        raw_model_response = "Error: No response from model"
    
    parsed_model_response = parse_model_response(raw_model_response)
    
    return {
        'run_id': run_id,
        'record_id': record_id,
        'datetime': run_datetime,
        'config': config,
        'prompt_template': prompt_template_str,
        'question_idx': q_data['original_idx'],
        'question': q_data['question'],
        'options': q_data['options'],
        'correct_idx': q_data['correct_idx'],
        'raw_model_response': raw_model_response,
        'parsed_model_response': parsed_model_response,
        'prompt': prompt,
        'token_usage': response.get('usage', {})
    }

def main():
    load_dotenv()    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path()
    config = get_config()
    
    print(f"Run ID: {run_id}")
    print(f"Datetime: {run_datetime}")
    print(f"Results will be appended to: {results_path}")
    
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    
    print(f"Selecting {NUM_QUESTIONS} questions with seed {RANDOM_SEED}")
    questions_data = select_questions_and_options(DATASET_NAME, dataset, NUM_QUESTIONS, NUM_CHOICES, RANDOM_SEED)
    
    prompt_template, response_format_prompt = load_prompt_template()

    key_info_start = get_openrouter_key_info(api_key)
    start_time = time.time()
    
    print(f"Processing {len(questions_data)} questions with {MAX_THREADS} threads...")
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_question, q_data, prompt_template, response_format_prompt, prompt_template, api_key, i+1, len(questions_data), config, run_id, run_datetime): i
            for i, q_data in enumerate(questions_data)
        }
        
        with open(results_path, 'a') as f:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    completed += 1
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    
    key_info_end = get_openrouter_key_info(api_key)
    duration = time.time() - start_time
    
    print(f"\n{completed}/{len(questions_data)} questions completed in {duration:.1f}s")
    print(f"Results: {results_path}")
    
    if key_info_start and key_info_end:
        start_usage = key_info_start.get('data', {}).get('usage', 0)
        end_usage = key_info_end.get('data', {}).get('usage', 0)
        print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")

if __name__ == "__main__":
    main()
