import os
import json
import yaml
import random
import string
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from config_verdict import (
    DEBATE_RUN_ID, JUDGE_MODEL, JUDGE_TEMPERATURE,
    JUDGE_REASONING_EFFORT, JUDGE_REASONING_MAX_TOKENS, MAX_OUTPUT_TOKENS
)
from llm_utils import call_openrouter, get_openrouter_key_info, parse_answer
from debate_utils import format_debate_history

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def setup_output_path(verdict_run_id):
    output_dir = Path('results') / 'verdicts'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{verdict_run_id}.jsonl'

def get_config():
    return {
        'debate_run_id': DEBATE_RUN_ID,
        'judge_model': JUDGE_MODEL,
        'judge_temperature': JUDGE_TEMPERATURE,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'judge_reasoning_effort': JUDGE_REASONING_EFFORT,
        'judge_reasoning_max_tokens': JUDGE_REASONING_MAX_TOKENS
    }

def load_prompts():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['judge_prompt_template'], prompts['response_format_prompt']

def run_judge(question, options, public_debate_history_text, judge_template, response_format_prompt, api_key, verdict_run_id, record_id):
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
        run_id=verdict_run_id,
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

def process_debate_record(debate_record, judge_template, response_format_prompt, judge_template_str, api_key, config, verdict_run_id, run_datetime):
    public_debate_history_text = format_debate_history(debate_record['debate_history'], show_private=False)
    
    judge_verdict = run_judge(
        debate_record['question'],
        debate_record['options'],
        public_debate_history_text,
        judge_template,
        response_format_prompt,
        api_key,
        verdict_run_id,
        debate_record['record_id']
    )
    
    return {
        'verdict_run_id': verdict_run_id,
        'debate_run_id': DEBATE_RUN_ID,
        'record_id': debate_record['record_id'],
        'datetime': run_datetime,
        'config': config,
        'prompt_template': judge_template_str,
        'question': debate_record['question'],
        'options': debate_record['options'],
        'correct_idx': debate_record['correct_idx'],
        'judge_verdict': judge_verdict
    }

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    verdict_run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path(verdict_run_id)
    config = get_config()
    
    print(f"Verdict Run ID: {verdict_run_id}")
    print(f"Debate Run ID: {DEBATE_RUN_ID}")
    print(f"Datetime: {run_datetime}")
    print(f"Results: {results_path}")
    
    debate_path = Path('results') / 'debate' / f'{DEBATE_RUN_ID}.jsonl'
    if not debate_path.exists():
        raise ValueError(f"Debate results not found: {debate_path}")
    
    debate_records = []
    with open(debate_path, 'r') as f:
        for line in f:
            debate_records.append(json.loads(line))
    
    judge_template, response_format_prompt = load_prompts()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    start_time = time.time()
    print(f"Processing {len(debate_records)} debates...")
    completed = 0
    
    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = {
            executor.submit(process_debate_record, debate_record, judge_template, response_format_prompt, judge_template, api_key, config, verdict_run_id, run_datetime): i
            for i, debate_record in enumerate(debate_records)
        }
        
        with open(results_path, 'w') as f:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    completed += 1
                    
                    key_info_current = get_openrouter_key_info(api_key)
                    current_usage = key_info_current.get('data', {}).get('usage', 0)
                    cost_so_far = current_usage - start_usage
                    print(f"Completed {completed}/{len(debate_records)} - Record ID: {result['record_id']} - Cost: ${cost_so_far:.6f}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue
    
    duration = time.time() - start_time
    print(f"\n{completed}/{len(debate_records)} verdicts completed in {duration:.1f}s")
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0)
    print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")

if __name__ == "__main__":
    main()

