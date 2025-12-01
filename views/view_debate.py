"""
Prints the debate history for a given run_id and record_id.
"""
import argparse
import json
import random
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.shared_utils import format_latex
from views.shared_view_utils import find_and_display_qa


from utils.debate_utils import format_debate_history

def display_debate(debate_data, hide_private=False, upto_turns=None, do_latex_formatting=False, view_qa=False):
    # if not debate_data.get('success', True):
    #     print(f"{'='*80}")
    #     print(f"ERROR: Debate failed")
    #     print(f"{'='*80}")
    #     print(f"Run ID: {debate_data.get('run_id')}")
    #     print(f"Record ID: {debate_data.get('record_id')}")
    #     # Find the turn with the error
    #     for turn in debate_data['debate_history']:
    #         if 'success' not in turn or not turn['success']:
    #             print(f"\n\nRaw Response: \n{turn['raw_response']}")
    #     print(f"\n\nTHE ERROR ASSOCIATED WITH THIS RECORD IS: \n{debate_data.get('error_message', 'Unknown error')}")
    #     return False
    
    if hide_private:
        show_private = False
    else:
        show_private = debate_data['config'].get('private_scratchpad', False)
    debate_text = format_debate_history(debate_data['debate_history'], show_private=show_private, upto_turns=upto_turns, do_latex_formatting=do_latex_formatting)
    
    print(f"{'='*80}")
    # print(f"Debate Run: {debate_data['run_id']} | Record: {debate_data['record_id']} | Question Idx: {debate_data['question_idx']} | Correct Idx: {debate_data['correct_idx']}")
    print(f"Debate Run: {debate_data['run_id']} | Record: {debate_data['record_id']} | Question Idx: {debate_data['question_idx']} | Dataset: {debate_data['config']['dataset_name']}")
    print(f"{'='*80}\nQuestion\n{'='*80}")
    print(format_latex(debate_data['question']))
    print(f"\nOptions: {[format_latex(opt) for opt in debate_data['options']]}")
    print(f"{'='*80}\nDebate\n{'='*80}")
    print(f"{debate_text}")
    if view_qa:
        find_and_display_qa(debate_data)
    return True

def load_debate_data(run_id, record_id=None, random_id=False):
    debate_path = project_root / 'results' / ('human/human_interactive_debate.jsonl' if run_id == 'human' else f'debates/{run_id}.jsonl')
    records = [json.loads(line) for line in open(debate_path)]
    if random_id:
        return random.choice(records) if records else None
    return next((r for r in records if r['record_id'] == record_id), None)

def main():
    parser = argparse.ArgumentParser(description='View debate results')
    parser.add_argument('run_id')
    parser.add_argument('record_id', nargs='?', help='Optional record ID to view specific record')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    parser.add_argument('--random_id', action='store_true', help='Pick a random record from the file')
    parser.add_argument('--view-qa', action='store_true', help='View the associated judge QA')
    args = parser.parse_args()
    
    debate_data = load_debate_data(args.run_id, record_id=args.record_id, random_id=args.random_id)
    if not debate_data:
        print(f"Record {args.record_id} not found in debate run {args.run_id}")
        return
    
    display_debate(debate_data, hide_private=args.hide_private, view_qa=args.view_qa, do_latex_formatting=True)

if __name__ == '__main__':
    main()