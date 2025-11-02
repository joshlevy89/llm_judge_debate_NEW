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
import config.config_verdict as config_verdict
from config.config_verdict import (
    DEBATE_RUN_ID, JUDGE_MODEL, JUDGE_TEMPERATURE,
    JUDGE_REASONING_EFFORT, JUDGE_REASONING_MAX_TOKENS, MAX_OUTPUT_TOKENS,
    SUBSET_N, SPECIFIC_RECORD_IDS, MAX_THREADS
)
from utils.llm_utils import call_openrouter, get_openrouter_key_info, parse_answer, log_progress
from utils.debate_utils import format_debate_history
from utils.shared_utils import extract_config

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def setup_output_path(verdict_run_id):
    output_dir = Path('results') / 'verdicts'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{verdict_run_id}.jsonl'

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
        'success': True,
        'error_message': None,
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
    config = extract_config(config_verdict)
    
    print(f"Verdict Run ID: {verdict_run_id}")
    print(f"Debate Run ID: {DEBATE_RUN_ID}")
    print(f"Datetime: {run_datetime}")
    print(f"Results: {results_path}")
    
    debate_path = Path('results') / 'debates' / f'{DEBATE_RUN_ID}.jsonl'
    if not debate_path.exists():
        raise ValueError(f"Debate results not found: {debate_path}")
    
    debate_records = []
    with open(debate_path, 'r') as f:
        for line in f:
            debate_records.append(json.loads(line))
    
    if SPECIFIC_RECORD_IDS is not None:
        record_id_set = set(SPECIFIC_RECORD_IDS)
        debate_records = [r for r in debate_records if r['record_id'] in record_id_set]
    elif SUBSET_N is not None:
        debate_records = debate_records[:SUBSET_N]
    
    judge_template, response_format_prompt = load_prompts()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    start_time = time.time()
    print(f"Processing {len(debate_records)} debates...")
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_debate_record, debate_record, judge_template, response_format_prompt, judge_template, api_key, config, verdict_run_id, run_datetime): debate_record
            for i, debate_record in enumerate(debate_records)
        }
        
        with open(results_path, 'w') as f:
            for future in as_completed(futures):
                debate_record = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    is_correct = result['judge_verdict']['parsed']['answer'] == result['correct_idx'] if result['judge_verdict']['parsed']['answer'] is not None else None
                    log_progress("completed", completed, len(debate_records), result['verdict_run_id'], result['record_id'], api_key, start_usage, is_correct=is_correct)
                except Exception as e:
                    failed += 1
                    result = {
                        'success': False,
                        'error_message': str(e),
                        'verdict_run_id': verdict_run_id,
                        'record_id': debate_record['record_id'],
                        'debate_run_id': DEBATE_RUN_ID,
                        'datetime': run_datetime,
                        'config': config,
                        'question': debate_record['question'],
                        'options': debate_record['options'],
                        'correct_idx': debate_record['correct_idx']
                    }
                    log_progress("failed", failed, len(debate_records), verdict_run_id, debate_record['record_id'], api_key, start_usage, error=str(e))
                
                f.write(json.dumps(result) + '\n')
                f.flush()
    
    duration = time.time() - start_time
    
    results_path = f"results/verdicts/{verdict_run_id}.jsonl"
    with open(results_path, 'r') as f:
        results = [json.loads(line) for line in f]
    null_count = sum(1 for r in results if r.get('success') and r.get('judge_verdict', {}).get('parsed', {}).get('answer') is None)
    
    print(f"\nRun ID: {verdict_run_id}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: total {len(debate_records)}, success {completed}, error {failed}, null {null_count}")
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0)
    print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")

if __name__ == "__main__":
    main()

