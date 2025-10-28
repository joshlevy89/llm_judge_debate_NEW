import os
import re
import json
import yaml
import random
import string
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
from qa_config import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    MODEL_NAME,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES,
    MAX_THREADS
)
from llm_utils import call_openrouter
from dataset_utils import select_questions_and_options, format_options

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def setup_output_paths(model_name, dataset_name):
    run_id = generate_run_id()
    
    # Clean model and dataset names for filename
    model_clean = model_name.replace('/', '_')
    dataset_clean = dataset_name.replace('/', '_')
    
    # Create directory structure
    output_dir = Path('results') / 'qa'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file paths
    base_name = f"{model_clean}_{dataset_clean}_{run_id}"
    results_path = output_dir / f"{base_name}.jsonl"
    config_path = output_dir / f"{base_name}.json"
    
    return results_path, config_path

def save_config(config_path):
    config = {
        'dataset_name': DATASET_NAME,
        'dataset_subset': DATASET_SUBSET,
        'dataset_split': DATASET_SPLIT,
        'model_name': MODEL_NAME,
        'num_questions': NUM_QUESTIONS,
        'random_seed': RANDOM_SEED,
        'num_choices': NUM_CHOICES,
        'max_threads': MAX_THREADS
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_prompt_template():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['qa_prompt_template']

def parse_model_response(response_text):
    parsed = {
        'is_valid': False,
        'answer': None,
        'confidence': None,
        'reasoning': None
    }
    
    final_answer_match = re.search(r'<BEGIN FINAL ANSWER>(.*?)</END FINAL ANSWER>', response_text, re.DOTALL | re.IGNORECASE)
    if not final_answer_match:
        return parsed
    
    final_answer_text = final_answer_match.group(1)
    parsed['is_valid'] = True
    
    answer_match = re.search(r'Answer:\s*(\d+)', final_answer_text, re.IGNORECASE)
    if answer_match:
        parsed['answer'] = int(answer_match.group(1))
    
    confidence_match = re.search(r'Confidence:\s*(\d+)(?:\.\d+)?%?', final_answer_text, re.IGNORECASE)
    if confidence_match:
        parsed['confidence'] = int(confidence_match.group(1))
    
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\s*$|\Z)', final_answer_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        parsed['reasoning'] = reasoning_match.group(1).strip()
    
    return parsed

def process_question(q_data, prompt_template, api_key, question_num, total_questions):
    print(f"Processing question {question_num}/{total_questions}")
    
    options_text = format_options(q_data['options'])
    number_choices = ', '.join(str(i) for i in range(NUM_CHOICES))
    
    prompt = prompt_template.format(
        question=q_data['question'],
        options_text=options_text,
        letter_choices=number_choices,
    )
    
    response = call_openrouter(prompt, MODEL_NAME, api_key)
    
    if 'choices' in response and len(response['choices']) > 0:
        raw_model_response = response['choices'][0]['message']['content']
    else:
        raw_model_response = "Error: No response from model"
    
    parsed_model_response = parse_model_response(raw_model_response)
    
    return {
        'question_idx': q_data['original_idx'],
        'question': q_data['question'],
        'options': q_data['options'],
        'correct_idx': q_data['correct_idx'],
        'raw_model_response': raw_model_response,
        'parsed_model_response': parsed_model_response,
        'prompt': prompt
    }

def main():
    load_dotenv()    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    # Setup output paths
    results_path, config_path = setup_output_paths(MODEL_NAME, DATASET_NAME)
    print(f"Results will be saved to: {results_path}")
    
    # Save config
    save_config(config_path)
    print(f"Config saved to: {config_path}")
    
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    
    print(f"Selecting {NUM_QUESTIONS} questions with seed {RANDOM_SEED}")
    questions_data = select_questions_and_options(DATASET_NAME, dataset, NUM_QUESTIONS, NUM_CHOICES, RANDOM_SEED)
    
    prompt_template = load_prompt_template()

    # Run evaluation in parallel
    print(f"Processing {len(questions_data)} questions with {MAX_THREADS} threads")
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_question, q_data, prompt_template, api_key, i+1, len(questions_data)): i
            for i, q_data in enumerate(questions_data)
        }
        
        with open(results_path, 'w') as f:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    completed += 1
                except Exception as e:
                    print(f"Error processing question: {e}")
                    continue
    
    print(f"Evaluation complete. {completed}/{len(questions_data)} questions processed.")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
