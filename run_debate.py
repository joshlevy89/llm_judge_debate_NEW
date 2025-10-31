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
from config_debate import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    DEBATER_MODEL, JUDGE_MODEL, DEBATER_TEMPERATURE, JUDGE_TEMPERATURE,
    DEBATER_REASONING_EFFORT, DEBATER_REASONING_MAX_TOKENS,
    JUDGE_REASONING_EFFORT, JUDGE_REASONING_MAX_TOKENS,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES, NUM_TURNS,
    PRIVATE_SCRATCHPAD, MAX_THREADS, MAX_OUTPUT_TOKENS
)
from llm_utils import call_openrouter, get_openrouter_key_info, parse_answer, parse_debater_response, log_llm_error
from dataset_utils import select_questions_and_options, format_options
from debate_utils import format_debate_history

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def setup_output_path(run_id):
    output_dir = Path('results') / 'debate'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{run_id}.jsonl'

def get_config():
    return {
        'dataset_name': DATASET_NAME,
        'dataset_subset': DATASET_SUBSET,
        'dataset_split': DATASET_SPLIT,
        'debater_model': DEBATER_MODEL,
        'judge_model': JUDGE_MODEL,
        'debater_temperature': DEBATER_TEMPERATURE,
        'judge_temperature': JUDGE_TEMPERATURE,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'debater_reasoning_effort': DEBATER_REASONING_EFFORT,
        'debater_reasoning_max_tokens': DEBATER_REASONING_MAX_TOKENS,
        'judge_reasoning_effort': JUDGE_REASONING_EFFORT,
        'judge_reasoning_max_tokens': JUDGE_REASONING_MAX_TOKENS,
        'num_questions': NUM_QUESTIONS,
        'random_seed': RANDOM_SEED,
        'num_choices': NUM_CHOICES,
        'num_turns': NUM_TURNS,
        'private_scratchpad': PRIVATE_SCRATCHPAD,
        'max_threads': MAX_THREADS
    }

def load_prompts():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['debater_prompt_template'], prompts['judge_prompt_template'], prompts['response_format_prompt'], prompts['private_reasoning_prompt']

def get_debater_prompt(debater_idx, my_answer, all_answers, question, history, debater_template, private_reasoning_prompt):
    opponents_text = "\nYour opponents are arguing for:\n"
    for i, ans in enumerate(all_answers):
        if i != debater_idx:
            opponents_text += f"- Debater {i}: {ans}\n"
    
    public_debate_history_text = format_debate_history(history, show_private=False) if history else "\nThis is the first turn of the debate."
    
    private_reasoning_text = private_reasoning_prompt if PRIVATE_SCRATCHPAD else ""
    
    return debater_template.format(
        role=debater_idx,
        question=question,
        my_answer=my_answer,
        opponents_arguing_for_text=opponents_text,
        public_debate_history_text=public_debate_history_text,
        private_reasoning_prompt=private_reasoning_text
    )

def run_debate_turn(turn_num, debater_assignments, question, history, debater_template, private_reasoning_prompt, api_key, run_id, record_id):
    turn_responses = []
    
    for debater_idx, answer in enumerate(debater_assignments):
        prompt = get_debater_prompt(debater_idx, answer, debater_assignments, question, history, debater_template, private_reasoning_prompt)
        context = f"Debater {debater_idx} Turn {turn_num}"
        
        response, token_usage = call_openrouter(
            prompt, 
            DEBATER_MODEL, 
            api_key, 
            DEBATER_TEMPERATURE,
            reasoning_effort=DEBATER_REASONING_EFFORT,
            reasoning_max_tokens=DEBATER_REASONING_MAX_TOKENS,
            max_tokens=MAX_OUTPUT_TOKENS,
            run_id=run_id,
            record_id=record_id,
            context=context
        )
        
        response_text = response['content']
        parsed_response, parse_error = parse_debater_response(response_text, PRIVATE_SCRATCHPAD)
        
        if parse_error:
            error_msg = f"{context} Parsing Error: {parse_error}\n\nRaw Response:\n{response_text}"
            log_llm_error(run_id, record_id, error_msg)
            return None

        turn_responses.append({
            'turn': turn_num,
            'debater_idx': debater_idx,
            'raw_response': response_text,
            'internal_model_reasoning': response.get('reasoning'),
            'internal_model_reasoning_details': response.get('reasoning_details'),
            'parsed_response': parsed_response,
            'token_usage': token_usage
        })
    
    return turn_responses

def run_judge(question, options, public_debate_history_text, judge_template, response_format_prompt, api_key, run_id, record_id):
    options_text = ""
    for i, option in enumerate(options):
        options_text += f"Debater {i} is arguing for: {option}\n"
    
    min_confidence = int(100 / len(options))
    
    prompt = judge_template.format(
        question=question,
        num_debaters=len(options),
        options_text=options_text.strip(),
        public_debate_history_text=public_debate_history_text,
        min_confidence=min_confidence,
        response_format_prompt=response_format_prompt
    )
    
    response, token_usage = call_openrouter(
        prompt, 
        JUDGE_MODEL, 
        api_key, 
        JUDGE_TEMPERATURE,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        reasoning_max_tokens=JUDGE_REASONING_MAX_TOKENS,
        max_tokens=MAX_OUTPUT_TOKENS,
        run_id=run_id,
        record_id=record_id,
        context="Judge"
    )
    
    response_text = response['content']
    parsed = parse_answer(response_text)
    
    return {
        'raw_response': response_text,
        'internal_model_reasoning': response.get('reasoning'),
        'internal_model_reasoning_details': response.get('reasoning_details'),
        'parsed': parsed,
        'prompt': prompt,
        'token_usage': token_usage
    }

def process_question(q_data, debater_template, judge_template, response_format_prompt, private_reasoning_prompt, debater_template_str, judge_template_str, api_key, config, run_id, run_datetime):
    record_id = generate_run_id()
    debater_assignments = q_data['options']
    debate_history = []
    
    for turn in range(NUM_TURNS):
        turn_responses = run_debate_turn(turn, debater_assignments, q_data['question'], debate_history, debater_template, private_reasoning_prompt, api_key, run_id, record_id)
        if turn_responses is None:
            return None
        debate_history.extend(turn_responses)
    
    public_debate_history_text = format_debate_history(debate_history, show_private=False)
    judge_verdict = run_judge(q_data['question'], q_data['options'], public_debate_history_text, judge_template, response_format_prompt, api_key, run_id, record_id)
    
    return {
        'run_id': run_id,
        'record_id': record_id,
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
        'judge_verdict': judge_verdict
    }

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path(run_id)
    config = get_config()
    
    print(f"Run ID: {run_id}")
    print(f"Datetime: {run_datetime}")
    print(f"Results: {results_path}")
    
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    questions_data = select_questions_and_options(DATASET_NAME, dataset, NUM_QUESTIONS, NUM_CHOICES, RANDOM_SEED)
    
    debater_template, judge_template, response_format_prompt, private_reasoning_prompt = load_prompts()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    start_time = time.time()
    print(f"Processing {len(questions_data)} questions...")
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_question, q_data, debater_template, judge_template, response_format_prompt, private_reasoning_prompt, debater_template, judge_template, api_key, config, run_id, run_datetime): i
            for i, q_data in enumerate(questions_data)
        }
        
        with open(results_path, 'w') as f:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is None:
                        continue
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    completed += 1
                    
                    key_info_current = get_openrouter_key_info(api_key)
                    current_usage = key_info_current.get('data', {}).get('usage', 0)
                    cost_so_far = current_usage - start_usage
                    print(f"Completed {completed}/{len(questions_data)} - Record ID: {result['record_id']} - Cost: ${cost_so_far:.6f}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue
    
    duration = time.time() - start_time
    print(f"\n{completed}/{len(questions_data)} questions completed in {duration:.1f}s")
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0)
    print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")

if __name__ == "__main__":
    main()

