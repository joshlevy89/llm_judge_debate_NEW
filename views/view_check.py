"""
View leak check results for a given run_id and record_id.
Shows the question, options, and leak analysis for each checked argument.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_check_data(run_id, record_id):
    check_path = Path('results') / 'debate_checks' / f'{run_id}.jsonl'
    if not check_path.exists():
        return None
    
    checks_dict = {}
    for line in open(check_path):
        data = json.loads(line)
        if data['record_id'] == record_id:
            key = (data['debater_idx'], data['turn_idx'])
            checks_dict[key] = data
    
    checks = list(checks_dict.values()) if checks_dict else None
    # Sort checks by the turn_idx and debater_idx
    checks.sort(key=lambda x: (x['turn_idx'], x['debater_idx']))

    return checks

def load_debate_data(run_id, record_id):
    debate_path = Path('results') / 'debates' / f'{run_id}.jsonl'
    if not debate_path.exists():
        return None
    for line in open(debate_path):
        data = json.loads(line)
        if data['record_id'] == record_id:
            return data
    return None

def display_check(check_data):
    if not check_data.get('success', True):
        print(f"\n{'='*80}\nERROR: Check failed\n{'='*80}")
        print(f"Run ID: {check_data['run_id']}")
        print(f"Record ID: {check_data['record_id']}")
        print(f"Debater {check_data['debater_idx']}, Turn {check_data['turn_idx']}")
        print(f"Error: {check_data.get('error', 'Unknown error')}")
        return
    
    parsed = check_data['parsed_response']
    
    print(f"\n{'='*80}")
    print(f"Public Argument: Debater {check_data['debater_idx']}, Turn {check_data['turn_idx']}")
    print(f"{'='*80}")

    print(check_data['public_argument'])
    
    print(f"\n{'='*80}")
    print(f"Leak Analysis")
    print(f"{'='*80}")
    
    print(parsed['analysis'])
    print(f"\nLeaks Detected: {parsed['leaks_detected']}")
    print(f"Leak Types: {parsed['leak_types']}")
    print(f"Description: {parsed['description']}")

def main():
    parser = argparse.ArgumentParser(description='View leak check results')
    parser.add_argument('run_id', help='Debate run ID')
    parser.add_argument('record_id', help='Record ID')
    args = parser.parse_args()
    
    check_data_list = load_check_data(args.run_id, args.record_id)
    if not check_data_list:
        print(f"No check results found for record {args.record_id} in run {args.run_id}")
        return
    
    debate_data = load_debate_data(args.run_id, args.record_id)

    print(f"{'='*80}")
    print(f"Run: {debate_data['run_id']} | Record: {debate_data['record_id']} | Question Idx: {debate_data['question_idx']} | Choice Idx: {debate_data['correct_idx']}")
    print(f"{'='*80}\nQuestion\n{'='*80}")
    print(f"{debate_data['question']}\n")
    print(f"Options: {debate_data['options']}")
    
    for check_data in check_data_list:
        display_check(check_data)
    
    print(f"\n{'='*80}")
    print(f"Summary: {len(check_data_list)} check(s) for record {args.record_id}")
    leaks = sum(1 for c in check_data_list if c.get('success') and c['parsed_response']['leaks_detected'])
    print(f"Leaks found: {leaks}/{len(check_data_list)}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

