"""
Prints the verdict results for a given verdict_run_id and record_id.
Shows the debate from the referenced debate_run_id and the verdict.
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from views.view_debate import display_debate, load_debate_data
from views.shared_view_utils import find_and_display_qa

def load_verdict_data(verdict_run_id, record_id, first_record=False):
    verdict_path = project_root / 'results' / 'verdicts' / f'{verdict_run_id}.jsonl'
    for line in open(verdict_path):
        data = json.loads(line)
        if first_record:
            return data
        if data['record_id'] == record_id:
            return data
    return None


def display_verdict(verdict_data, debate_data):
    print(verdict_data['config'].keys())
    print(f"{'='*80}\nJudge Verdict (Verdict Run: {verdict_data['verdict_run_id']} | Question Idx: {debate_data['question_idx']} | Judge Model: {verdict_data['config']['judge_model']})\n{'='*80}")
    if verdict_data['judge_verdict'].get('internal_model_reasoning') is not None:
        print(f"[BEGIN INTERNAL REASONING]\n{verdict_data['judge_verdict']['internal_model_reasoning']}\n[END INTERNAL REASONING]\n")
    print(f"[BEGIN RAW RESPONSE]\n{verdict_data['judge_verdict']['raw_response']}\n[END RAW RESPONSE]")
    print(f"{'-'*80}\nJudge Parsed Response\n{'-'*80}")
    parsed_response = verdict_data['judge_verdict']['parsed']
    print(f"Answer: {parsed_response['answer']}")
    print(f"Confidence: {parsed_response['confidence']}")
    print(f"Reasoning: {parsed_response['reasoning']}")
    print(f"{'-'*80}\nOutput\n{'-'*80}")
    print(f"Correct Answer: {verdict_data['correct_idx']}")
    print(f"Judge Verdict: {parsed_response['answer']}")
    print(f"Correct: {parsed_response['answer'] == verdict_data['correct_idx']}")
    print()


def display_full_verdict(verdict_run_id, record_id, hide_private=False, view_qa=False, first_record=False):
    verdict_data = load_verdict_data(verdict_run_id, record_id, first_record=first_record)
    if not verdict_data:
        if first_record:
            print(f"No records found in verdict run {verdict_run_id}")
        else:
            print(f"Record {record_id} not found in verdict run {verdict_run_id}")
        return False
    
    if first_record:
        record_id = verdict_data['record_id']
    
    if not verdict_data.get('success', True):
        print(f"{'='*80}")
        print(f"ERROR: Verdict failed")
        print(f"{'='*80}")
        print(f"Verdict Run ID: {verdict_run_id}")
        print(f"Record ID: {record_id}")
        print(f"\nThe error that occurred during the run:\n\n{verdict_data.get('error_message', 'Unknown error')}")
        return False
    
    debate_data = load_debate_data(verdict_data['debate_run_id'], record_id)
    if not debate_data:
        print(f"Record {record_id} not found in debate run {verdict_data['debate_run_id']}")
        return False
    
    upto_turns = verdict_data['config'].get('upto_turns')
    success = display_debate(debate_data, hide_private=hide_private, upto_turns=upto_turns)
    if not success:
        return False
    
    display_verdict(verdict_data, debate_data)    
    if view_qa:
        find_and_display_qa(verdict_data)
        find_and_display_qa(debate_data)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='View verdict results')
    parser.add_argument('verdict_run_id')
    parser.add_argument('record_id', nargs='?', help='Record ID (optional if --first-record is used)')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    parser.add_argument('--view-qa', action='store_true', help='View the associated judge QA')
    parser.add_argument('--first-record', action='store_true', help='Use the first record instead of record_id')
    args = parser.parse_args()
    
    display_full_verdict(args.verdict_run_id, args.record_id, 
                        hide_private=args.hide_private, 
                        view_qa=args.view_qa, first_record=args.first_record)

if __name__ == '__main__':
    main()

