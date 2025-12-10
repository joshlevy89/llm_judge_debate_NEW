"""
Check debate arguments for correct answer leaks.

Analyzes debates where debaters are assigned incorrect answers to detect cases where
they inadvertently reveal awareness that their position is problematic.
"""
import os
import json
import re
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import config.config_check as config_check
from config.config_check import MODEL_NAME, TEMPERATURE, MAX_THREADS #, LEAK_TYPES
from utils.llm_utils import call_openrouter, get_openrouter_key_info
from utils.shared_utils import extract_config, load_prompts

load_dotenv()

# def format_leak_types():
#     return '\n'.join(f"- Type {k}: {v}" for k, v in LEAK_TYPES.items())

def format_options(options):
    return '\n'.join(f"{i}. {opt}" for i, opt in enumerate(options))

def parse_llm_response(response_text):
    analysis_match = re.search(r'<BEGIN ANALYSIS>(.*?)</END ANALYSIS>', response_text, re.DOTALL | re.IGNORECASE)
    issues_match = re.search(r'Leaks Detected:\s*(True|False)', response_text, re.IGNORECASE)
    types_match = re.search(r'Types of Leaks Detected:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    desc_match = re.search(r'Description of Leak\(s\):\s*(.+)', response_text, re.DOTALL | re.IGNORECASE)
    
    return {
        'analysis': analysis_match.group(1).strip() if analysis_match else None,
        'leaks_detected': issues_match.group(1).lower() == 'true' if issues_match else None,
        'leak_types': types_match.group(1).strip() if types_match else None,
        'description': desc_match.group(1).strip() if desc_match else None
    }

def process_turn(record_item, turn_data, prompt_template, api_key):
    debater_idx = turn_data['debater_idx']
    turn_idx = turn_data['turn_idx']
    public_argument = turn_data['public_argument']
    
    try:
        prompt = prompt_template.format(
            # leak_types=format_leak_types(),
            question=record_item['question'],
            options_str=format_options(record_item['options']),
            correct_idx=record_item['correct_idx'],
            debater_idx=debater_idx,
            public_argument=public_argument
        )
        
        response, _ = call_openrouter(
            prompt=prompt,
            model_name=MODEL_NAME,
            api_key=api_key,
            temperature=TEMPERATURE
        )
        
        parsed = parse_llm_response(response['content'])
        
        return {
            'debater_idx': debater_idx,
            'turn_idx': turn_idx,
            'public_argument': public_argument,
            'prompt': prompt,
            'raw_response': response['content'],
            'parsed_response': parsed
        }
    except Exception as e:
        return {
            'debater_idx': debater_idx,
            'turn_idx': turn_idx,
            'public_argument': public_argument,
            'error': str(e)
        }

def process_record(record_item, prompt_template, api_key):
    run_id = record_item['run_id']
    record_id = record_item['record_id']
    question = record_item['question']
    options = record_item['options']
    correct_idx = record_item['correct_idx']
    turns_data = record_item['turns']
    
    turns_results = []
    debate_has_leak = False
    config = extract_config(config_check)
    
    try:
        for turn_data in turns_data:
            turn_result = process_turn(record_item, turn_data, prompt_template, api_key)
            turns_results.append(turn_result)
            
            if 'error' not in turn_result and turn_result.get('parsed_response', {}).get('leaks_detected'):
                debate_has_leak = True
        
        return {
            'success': True,
            'run_id': run_id,
            'record_id': record_id,
            'question': question,
            'options': options,
            'correct_idx': correct_idx,
            'prompt_template': prompt_template,
            # 'leak_types': LEAK_TYPES,
            'config': config,
            'has_leak': debate_has_leak,
            'turns': turns_results
        }
    except Exception as e:
        return {
            'success': False,
            'run_id': run_id,
            'record_id': record_id,
            'question': question,
            'options': options,
            'correct_idx': correct_idx,
            'prompt_template': prompt_template,
            # 'leak_types': LEAK_TYPES,
            'config': config,
            'error': str(e),
            'turns': turns_results
        }

def check_debate_for_leaks(run_id, api_key, record_ids=None, filter_turn=None, filter_debater_idx=None):
    debate_file = Path('results') / 'debates' / f'{run_id}.jsonl'
    if not debate_file.exists():
        print(f"Debate file not found: {debate_file}")
        return
    
    output_dir = Path('results') / 'debate_checks'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{run_id}.jsonl'
    
    with open(debate_file) as f:
        debates = [json.loads(line) for line in f]
    
    if record_ids:
        record_ids_set = set(record_ids)
        debates = [d for d in debates if d.get('record_id') in record_ids_set]
    
    filters = [
        f"record_ids={','.join(record_ids)}" if record_ids else None,
        f"turn={filter_turn}" if filter_turn is not None else None,
        f"debater_idx={filter_debater_idx}" if filter_debater_idx is not None else None
    ]
    filters = [f for f in filters if f]
    
    if filters:
        print(f"Loaded {len(debates)} debate records (filtered: {', '.join(filters)})")
    else:
        print(f"Loaded {len(debates)} debate records")
    
    record_items = []
    num_debates_with_errors = 0 
    for debate in debates:
        if not debate.get('success'):
            num_debates_with_errors += 1
            continue
        
        record_id = debate['record_id']
        question = debate['question']
        options = debate['options']
        correct_idx = debate['correct_idx']
        
        turns = []
        for turn in debate.get('debate_history', []):
            debater_idx = turn['debater_idx']
            turn_idx = turn['turn']
            
            if filter_turn is not None and turn_idx != filter_turn:
                continue
            
            if filter_debater_idx is not None and debater_idx != filter_debater_idx:
                continue
            
            if debater_idx == correct_idx:
                continue
            
            # public_argument = turn.get('parsed_response', {}).get('public_argument', '')
            public_argument = turn['raw_response']
            if not public_argument:
                continue
            
            turns.append({
                'debater_idx': debater_idx,
                'turn_idx': turn_idx,
                'public_argument': public_argument
            })
        
        if turns:
            record_items.append({
                'run_id': run_id,
                'record_id': record_id,
                'question': question,
                'options': options,
                'correct_idx': correct_idx,
                'turns': turns
            })
    
    print(f"Filtered out {num_debates_with_errors} debates with errors")
    print(f"Processing {len(record_items)} records with {MAX_THREADS} max threads")
    
    prompt_template = load_prompts('leak')
    
    key_info = get_openrouter_key_info(api_key)
    start_usage = key_info.get('data', {}).get('usage', 0) if key_info else 0
    
    checked = 0
    leaks_found = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_record, item, prompt_template, api_key): item for item in record_items}
        
        for future in as_completed(futures):
            result = future.result()
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
            checked += 1
            
            key_info = get_openrouter_key_info(api_key)
            current_usage = key_info.get('data', {}).get('usage', 0) if key_info else 0
            cost = current_usage - start_usage if current_usage else 0
            
            if result['success']:
                if result.get('has_leak'):
                    leaks_found += 1
                    status = "Leak found"
                else:
                    status = "Healthy"
                num_turns = len(result.get('turns', []))
                print(f"✓ {status} - Run {result['run_id']}, Record {result['record_id']} ({num_turns} turns) - {checked}/{len(record_items)} - Cost: ${cost:.6f}")
            else:
                print(f"✗ Error - Run {result['run_id']}, Record {result['record_id']}: {result.get('error', 'Unknown error')}")
    
    key_info = get_openrouter_key_info(api_key)
    final_usage = key_info.get('data', {}).get('usage', 0) if key_info else 0
    total_cost = final_usage - start_usage if final_usage else 0
    
    print(f"\n{'='*60}")
    print(f"Completed: Checked {checked} records")
    print(f"Records with leaks: {leaks_found}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check debate arguments for correct answer leaks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_checks.py q6wpwb7
  python run_checks.py q6wpwb7 --record_ids rec1 rec2 rec3
  python run_checks.py q6wpwb7 --turn 0
  python run_checks.py q6wpwb7 --debater_idx 2
  python run_checks.py q6wpwb7 --record_ids rec1 rec2 --turn 0 --debater_idx 2
        '''
    )
    
    parser.add_argument('run_id', help='Debate run ID')
    parser.add_argument('--record_ids', nargs='+', help='Filter to specific record IDs')
    parser.add_argument('--turn', type=int, help='Filter to specific turn')
    parser.add_argument('--debater_idx', type=int, help='Filter to specific debater index')
    
    args = parser.parse_args()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        exit(1)
    
    check_debate_for_leaks(
        args.run_id,
        api_key,
        args.record_ids,
        args.turn,
        args.debater_idx
    )
