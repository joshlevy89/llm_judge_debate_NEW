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
from debate_config import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    DEBATER_MODEL, JUDGE_MODEL, DEBATER_TEMPERATURE, JUDGE_TEMPERATURE,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES, NUM_TURNS,
    MAX_THREADS
)
from llm_utils import call_openrouter, get_openrouter_key_info, parse_model_response
from dataset_utils import select_questions_and_options, format_options

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def setup_output_path(run_id):
    output_dir = Path('results') / 'debate' / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_config():
    return {
        'dataset_name': DATASET_NAME,
        'dataset_subset': DATASET_SUBSET,
        'dataset_split': DATASET_SPLIT,
        'debater_model': DEBATER_MODEL,
        'judge_model': JUDGE_MODEL,
        'debater_temperature': DEBATER_TEMPERATURE,
        'judge_temperature': JUDGE_TEMPERATURE,
        'num_questions': NUM_QUESTIONS,
        'random_seed': RANDOM_SEED,
        'num_choices': NUM_CHOICES,
        'num_turns': NUM_TURNS,
        'max_threads': MAX_THREADS
    }

def load_prompts():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['debater_prompt_template'], prompts['judge_prompt_template']

def format_debate_history(history):
    if not history:
        return ""
    
    text = "\n\n"
    for entry in history:
        text += f"Debater {entry['debater_idx']} (Turn {entry['turn']}):\n{entry['response']}\n\n"
    return text

def get_debater_prompt(debater_idx, my_answer, all_answers, question, history, debater_template):
    opponents_text = "\nYour opponents are arguing for:\n"
    for i, ans in enumerate(all_answers):
        if i != debater_idx:
            opponents_text += f"- Debater {i}: {ans}\n"
    
    debate_history_text = format_debate_history(history) if history else "\nThis is the first turn of the debate."
    
    return debater_template.format(
        role=debater_idx,
        question=question,
        my_answer=my_answer,
        opponents_arguing_for_text=opponents_text,
        debate_history=debate_history_text
    )

def run_debate_turn(turn_num, debater_assignments, question, history, debater_template, api_key):
    turn_responses = []
    
    for debater_idx, answer in enumerate(debater_assignments):
        prompt = get_debater_prompt(debater_idx, answer, debater_assignments, question, history, debater_template)
        response = call_openrouter(prompt, DEBATER_MODEL, api_key, DEBATER_TEMPERATURE)
        
        if 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['message']['content']
        else:
            response_text = "Error: No response from model"
        
        turn_responses.append({
            'turn': turn_num,
            'debater_idx': debater_idx,
            'response': response_text,
            'token_usage': response.get('usage', {})
        })
    
    return turn_responses

def run_judge(question, options, debate_history_text, judge_template, api_key):
    options_text = ""
    for i, option in enumerate(options):
        options_text += f"Debater {i} is arguing for: {option}\n"
    
    min_confidence = int(100 / len(options))
    
    prompt = judge_template.format(
        question=question,
        num_debaters=len(options),
        options_text=options_text.strip(),
        debate_history=debate_history_text,
        min_confidence=min_confidence
    )
    
    response = call_openrouter(prompt, JUDGE_MODEL, api_key, JUDGE_TEMPERATURE)
    
    if 'choices' in response and len(response['choices']) > 0:
        response_text = response['choices'][0]['message']['content']
    else:
        response_text = "Error: No response from model"
    
    parsed = parse_model_response(response_text)
    
    return {
        'raw_response': response_text,
        'parsed': parsed,
        'prompt': prompt,
        'token_usage': response.get('usage', {})
    }

def process_question(q_data, debater_template, judge_template, debater_template_str, judge_template_str, api_key, config, run_id, run_datetime):
    debater_assignments = q_data['options']
    debate_history = []
    
    for turn in range(NUM_TURNS):
        turn_responses = run_debate_turn(turn, debater_assignments, q_data['question'], debate_history, debater_template, api_key)
        debate_history.extend(turn_responses)
    
    debate_history_text = format_debate_history(debate_history)
    judge_verdict = run_judge(q_data['question'], q_data['options'], debate_history_text, judge_template, api_key)
    
    return {
        'run_id': run_id,
        'datetime': run_datetime,
        'config': config,
        'prompt_templates': {
            'debater': debater_template_str,
            'judge': judge_template_str
        },
        'question_idx': q_data['original_idx'],
        'question': q_data['question'],
        'options': q_data['options'],
        'correct_idx': q_data['correct_idx'],
        'debate_history': debate_history,
        'debate_history_text': debate_history_text,
        'judge_verdict': judge_verdict
    }

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    output_dir = setup_output_path(run_id)
    config = get_config()
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run ID: {run_id}")
    print(f"Datetime: {run_datetime}")
    print(f"Results: {output_dir}")
    
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    questions_data = select_questions_and_options(DATASET_NAME, dataset, NUM_QUESTIONS, NUM_CHOICES, RANDOM_SEED)
    
    debater_template, judge_template = load_prompts()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_time = time.time()
    
    print(f"Processing {len(questions_data)} questions with {MAX_THREADS} threads...")
    completed = 0
    
    results_path = output_dir / 'results.jsonl'
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_question, q_data, debater_template, judge_template, debater_template, judge_template, api_key, config, run_id, run_datetime): i
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
                    print(f"Error: {e}")
                    continue
    
    key_info_end = get_openrouter_key_info(api_key)
    duration = time.time() - start_time
    
    print(f"\n{completed}/{len(questions_data)} questions completed in {duration:.1f}s")
    
    if key_info_start and key_info_end:
        start_usage = key_info_start.get('data', {}).get('usage', 0)
        end_usage = key_info_end.get('data', {}).get('usage', 0)
        print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")

if __name__ == "__main__":
    main()

