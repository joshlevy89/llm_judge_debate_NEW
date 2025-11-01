"""
Prints the verdict results for a given verdict_run_id and record_id.
Shows the debate from the referenced debate_run_id and the verdict.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from views.view_debate import display_debate, load_debate_data

def load_verdict_data(verdict_run_id, record_id):
    verdict_path = Path('results') / 'verdicts' / f'{verdict_run_id}.jsonl'
    for line in open(verdict_path):
        data = json.loads(line)
        if data['record_id'] == record_id:
            return data
    return None

def display_verdict(verdict_data):
    print(f"{'='*80}\nJudge (Verdict Run: {verdict_data['verdict_run_id']})\n{'='*80}")
    if verdict_data['judge_verdict'].get('internal_model_reasoning') is not None:
        print(f"[BEGIN INTERNAL REASONING]\n{verdict_data['judge_verdict']['internal_model_reasoning']}\n[END INTERNAL REASONING]\n")
    print(f"[BEGIN RAW RESPONSE]\n{verdict_data['judge_verdict']['raw_response']}\n[END RAW RESPONSE]")
    print(f"{'='*80}\nJudge Parsed Response\n{'='*80}")
    parsed_response = verdict_data['judge_verdict']['parsed']
    print(f"Answer: {parsed_response['answer']}")
    print(f"Confidence: {parsed_response['confidence']}")
    print(f"Reasoning: {parsed_response['reasoning']}")
    print(f"{'='*80}\nOutput\n{'='*80}")
    print(f"Correct Answer: {verdict_data['correct_idx']}")
    print(f"Judge Verdict: {parsed_response['answer']}")
    print(f"Correct: {parsed_response['answer'] == verdict_data['correct_idx']}")

def main():
    parser = argparse.ArgumentParser(description='View verdict results')
    parser.add_argument('verdict_run_id')
    parser.add_argument('record_id')
    parser.add_argument('--hide-private', action='store_true', help='Hide private reasoning')
    args = parser.parse_args()
    
    verdict_data = load_verdict_data(args.verdict_run_id, args.record_id)
    if not verdict_data:
        print(f"Record {args.record_id} not found in verdict run {args.verdict_run_id}")
        return
    
    debate_data = load_debate_data(verdict_data['debate_run_id'], args.record_id)
    if not debate_data:
        print(f"Record {args.record_id} not found in debate run {verdict_data['debate_run_id']}")
        return
    
    display_debate(debate_data, hide_private=args.hide_private)
    print()
    display_verdict(verdict_data)

if __name__ == '__main__':
    main()

