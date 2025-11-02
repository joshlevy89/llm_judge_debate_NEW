"""
Prints the debate history for a given run_id and record_id.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.debate_utils import format_debate_history

def display_debate(debate_data, hide_private=False):
    if not debate_data.get('success', True):
        print(f"{'='*80}")
        print(f"ERROR: Debate failed")
        print(f"{'='*80}")
        print(f"Run ID: {debate_data.get('run_id')}")
        print(f"Record ID: {debate_data.get('record_id')}")
        print(f"Error message: {debate_data.get('error_message', 'Unknown error')}")
        return False
    
    if hide_private:
        show_private = False
    else:
        show_private = debate_data['config'].get('private_scratchpad', False)
    debate_text = format_debate_history(debate_data['debate_history'], show_private=show_private)
    
    print(f"{'='*80}")
    print(f"Debate Run: {debate_data['run_id']} | Record: {debate_data['record_id']} | Question Idx: {debate_data['question_idx']}")
    print(f"{'='*80}\nQuestion\n{'='*80}")
    print(f"{debate_data['question']}\n")
    print(f"Options: {debate_data['options']}")
    print(f"{'='*80}\nDebate\n{'='*80}")
    print(f"{debate_text}")
    return True

def load_debate_data(run_id, record_id):
    debate_path = Path('results') / 'debates' / f'{run_id}.jsonl'
    for line in open(debate_path):
        data = json.loads(line)
        if data['record_id'] == record_id:
            return data
    return None

def main():
    parser = argparse.ArgumentParser(description='View debate results')
    parser.add_argument('run_id')
    parser.add_argument('record_id')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    args = parser.parse_args()
    
    debate_data = load_debate_data(args.run_id, args.record_id)
    if not debate_data:
        print(f"Record {args.record_id} not found in debate run {args.run_id}")
        return
    
    display_debate(debate_data, hide_private=args.hide_private)

if __name__ == '__main__':
    main()