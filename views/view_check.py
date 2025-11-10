"""
View leak check results for a given run_id and record_id.
Shows the question, options, and leak analysis for each checked argument.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent

def load_check_data(run_id, record_id):
    check_path = project_root / 'results' / 'debate_checks' / f'{run_id}.jsonl'
    check_df = pd.read_json(check_path, lines=True)
    print(check_df.columns)
    check_df = check_df.drop_duplicates(['run_id', 'record_id'], keep='last')
    match = check_df[check_df['record_id'] == record_id].iloc[0].to_dict()
    return match

def display_check(check_data):
    if not check_data.get('success', True):
        print(f"{'='*80}")
        print(f"ERROR: Check failed")
        print(f"{'='*80}")
        print(f"Run ID: {check_data['run_id']}")
        print(f"Record ID: {check_data['record_id']}")
        print(f"Error: {check_data.get('error', 'Unknown error')}")
        return
    
    print(f"{'='*80}")
    print(f"Run: {check_data['run_id']} | Record: {check_data['record_id']} | Correct Idx: {check_data['correct_idx']}")
    print(f"{'='*80}\nQuestion\n{'='*80}")
    print(f"{check_data['question']}\n")
    print(f"Options: {check_data['options']}")
    
    turns = check_data.get('turns', [])
    turns.sort(key=lambda x: (x['turn_idx'], x['debater_idx']))
    
    for turn in turns:
        if 'error' in turn:
            print(f"\n{'='*80}")
            print(f"ERROR: Turn check failed - Debater {turn['debater_idx']}, Turn {turn['turn_idx']}")
            print(f"{'='*80}")
            print(f"Error: {turn['error']}")
            continue
        
        parsed = turn.get('parsed_response', {})
        
        print(f"\n{'='*80}")
        print(f"Public Argument: Debater {turn['debater_idx']}, Turn {turn['turn_idx']}")
        print(f"{'='*80}")
        print(turn['public_argument'])
        
        print(f"\n{'='*80}")
        print(f"Leak Analysis")
        print(f"{'='*80}")
        if parsed.get('analysis'):
            print(parsed['analysis'])
        print(f"\nLeaks Detected: {parsed.get('leaks_detected')}")
        print(f"Leak Types: {parsed.get('leak_types')}")
        print(f"Description: {parsed.get('description')}")
    
    print(f"\n{'='*80}")
    print(f"Summary: {len(turns)} turn(s) checked")
    leaks = sum(1 for t in turns if 'error' not in t and t.get('parsed_response', {}).get('leaks_detected'))
    print(f"Leaks found: {leaks}/{len(turns)}")
    print(f"Has leak: {check_data.get('has_leak', False)}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='View leak check results')
    parser.add_argument('run_id')
    parser.add_argument('record_id')
    args = parser.parse_args()
    
    check_data = load_check_data(args.run_id, args.record_id)
    print(check_data)
    if not check_data:
        print(f"Record {args.record_id} not found in check run {args.run_id}")
        return
    
    display_check(check_data)

if __name__ == '__main__':
    main()

